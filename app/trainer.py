"""
Demo-scale PPO training for the app's Train tab.

This is deliberately NOT the training pipeline that produced the shipped
checkpoints. Those are 100-200M timestep runs on a GPU; the models here are
thousands of timesteps on CPU and exist to demonstrate the
train -> save -> select -> simulate loop end to end. A model trained here will
read badly, and the UI says so.

The real pipeline lives in sub_models/rl.py and is left untouched -- it selects
its level by commenting lines in the source, which suits a research workflow but
cannot be driven from a UI. We reuse its policy architecture (imported, not
copied, so the two cannot drift) and its env classes.
"""
import os
import re
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import model_params
from model_registry import SESSION_MODELS_DIR, LEVEL_LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "step5" / "simulators" / "simulator_v20250604"

# rl.py uses bare env imports (it is normally run from inside sub_models), and
# reaches for step5.utils.constants, so both roots have to be importable.
for p in (str(REPO_ROOT), str(SIM_DIR), str(SIM_DIR / "sub_models")):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore")

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Same feature extractor and LR schedule the real pipeline uses.
from rl import StatefulInformationExtractor, linear_schedule

from sub_models.word_recognition_v0807.WordRecognitionEnv import WordRecognitionEnv
from sub_models.sentence_read_v0604.SentenceReadingEnv import SentenceReadingUnderTimePressureEnv
from sub_models.text_read_v0604.TextReadEnv import TextReadingUnderTimePressureEnv
from sub_models.text_read_v0604.Utilities import DictActionUnwrapper


# Ceiling on what the UI will start, so nobody kicks off a run that outlives the
# Space. Demo-scale by construction.
MAX_TIMESTEPS = 200_000
DEFAULT_TIMESTEPS = 20_000

CONFIG_PATH = SIM_DIR / "sub_models" / "config.yaml"

# The envs decide train-vs-simulate behaviour by reading rl.mode out of
# config.yaml at construction time, from a path fixed relative to their own
# source file -- so it cannot be redirected per call. Training therefore has to
# flip that value on disk and put it back. This lock keeps a training run from
# overlapping a simulation that would otherwise read the flipped value.
_MODE_LOCK = threading.RLock()


@contextmanager
def _training_mode(level):
    """
    Temporarily switch the shared config to 'train' mode for the duration of a run.

    Also patches TextReadEnv.DATA_SOURCE, which is a module-level constant the
    text reader asserts on when training (it wants generated rather than real
    stimuli). Both are restored on the way out, including on error.
    """
    with _MODE_LOCK:
        original = CONFIG_PATH.read_text()
        patched, n = re.subn(
            r"^(\s*mode:\s*)\S+", r"\1train", original, count=1, flags=re.M
        )
        if n != 1:
            raise RuntimeError(f"Could not find 'mode:' to patch in {CONFIG_PATH}")

        text_env_module = None
        original_data_source = None
        try:
            CONFIG_PATH.write_text(patched)

            if level == "text_reader":
                from sub_models.text_read_v0604 import TextReadEnv as text_env_module
                original_data_source = text_env_module.DATA_SOURCE
                text_env_module.DATA_SOURCE = "generated_stimuli"

            yield
        finally:
            CONFIG_PATH.write_text(original)
            if text_env_module is not None and original_data_source is not None:
                text_env_module.DATA_SOURCE = original_data_source


class _ResetParams(gym.Wrapper):
    """
    Supplies the same `params` dict to every env.reset().

    The envs take their per-episode tunables through reset(params=...), which is
    how the Simulate path sets them. SB3 resets with no arguments, so without
    this the training envs would always fall back to their built-in defaults and
    the paper parameters would be unreachable from the Train tab.
    """

    def __init__(self, env, params):
        super().__init__(env)
        self._params = dict(params)

    def reset(self, **kwargs):
        kwargs.setdefault("params", self._params)
        return self.env.reset(**kwargs)


def _make_env(level, reset_params=None):
    """Build one env for `level`, matching how rl.py constructs it."""
    if level == "word_recognizer":
        env = WordRecognitionEnv()
    elif level == "sentence_reader":
        env = SentenceReadingUnderTimePressureEnv()
    elif level == "text_reader":
        env = DictActionUnwrapper(TextReadingUnderTimePressureEnv())
    else:
        raise ValueError(f"Unknown level '{level}'.")

    # Wrapped inside Monitor so the episode statistics still come from the
    # outermost wrapper, and outside DictActionUnwrapper so reset() reaches the
    # env that reads the params.
    if reset_params:
        env = _ResetParams(env, reset_params)
    return Monitor(env)


class _ProgressCallback(BaseCallback):
    """Feeds step counts and rolling episode reward back to the UI."""

    def __init__(self, total_timesteps, report):
        super().__init__()
        self._total = total_timesteps
        self._report = report
        self.reward_curve = []   # (timestep, mean_reward)
        self._last_report = 0.0

    def _on_step(self):
        # Monitor stores finished episodes on the underlying env buffers.
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf:
            mean_r = float(np.mean([ep["r"] for ep in buf]))
            self.reward_curve.append((int(self.num_timesteps), mean_r))

        now = time.time()
        if self._report and (now - self._last_report > 2.0):
            self._last_report = now
            frac = self.num_timesteps / max(self._total, 1)
            recent = self.reward_curve[-1][1] if self.reward_curve else float("nan")
            self._report(frac, self.num_timesteps, recent)
        return True


def _plot_curve(curve, level, out_path):
    """Training curve for the UI. Returns the path, or None when nothing to plot."""
    if not curve:
        return None
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(xs, ys, lw=1.5)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title(f"Demo training - {LEVEL_LABELS.get(level, level)}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def train_demo_model(
    level,
    total_timesteps=DEFAULT_TIMESTEPS,
    n_envs=2,
    learning_rate=1e-4,
    gamma=0.90,
    n_steps=512,
    batch_size=128,
    seed=42,
    run_name=None,
    model_parameters=None,
    report=None,
):
    """
    Train a small PPO model for one level and save it into the session store.

    `model_parameters` is a {key: value} dict of the paper's tunable parameters
    (see model_params); anything not applicable to `level` is dropped. Whatever
    survives is saved beside the checkpoint, because a checkpoint trained under
    non-default parameters is only meaningful when simulated under them too.

    Returns a dict: {model_id, model_path, folder, curve_png, summary, level}.
    `report(frac, steps, mean_reward)` is called periodically if supplied.
    """
    if total_timesteps > MAX_TIMESTEPS:
        raise ValueError(
            f"{total_timesteps} timesteps exceeds the {MAX_TIMESTEPS} demo ceiling."
        )

    applied_params = model_params.normalise(model_parameters, level=level)
    reset_params = model_params.reset_params(applied_params, level)

    # The envs read their assets with paths relative to the simulator directory.
    prev_cwd = os.getcwd()
    os.chdir(SIM_DIR)
    try:
        # The envs latch rl.mode at construction, so they must be built inside
        # the training-mode window; training itself does not depend on it.
        with _training_mode(level):
            # The "constant" parameters are latched by the envs in __init__, so
            # the patch only has to cover construction -- but it has to cover it
            # for every env in the vector, hence inside the DummyVecEnv call.
            with model_params.constants_applied(applied_params, level=level):
                # DummyVecEnv, not SubprocVecEnv: a free Space has ~2 vCPU, and
                # subprocess startup dominates at these run lengths.
                venv = DummyVecEnv([
                    (lambda: _make_env(level, reset_params))
                    for _ in range(max(1, n_envs))
                ])

        policy_kwargs = dict(
            features_extractor_class=StatefulInformationExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            activation_fn=th.nn.LeakyReLU,
            net_arch=[512, 512],
            log_std_init=-1.0,
            normalize_images=False,
        )

        model = PPO(
            policy="MlpPolicy",
            env=venv,
            verbose=0,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=linear_schedule(
                initial_value=float(learning_rate),
                min_value=float(learning_rate) / 100.0,
                threshold=0.8,
            ),
            gamma=gamma,
            device="cpu",   # no GPU on a free Space; MPS gives no win at this size
            seed=seed,
        )

        cb = _ProgressCallback(total_timesteps, report)
        started = time.time()
        model.learn(total_timesteps=total_timesteps, callback=cb)
        elapsed = time.time() - started

        stamp = time.strftime("%m%d_%H%M%S")
        folder_name = run_name.strip() if run_name and run_name.strip() else f"demo_{level}_{stamp}"
        folder = SESSION_MODELS_DIR / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        # Same naming the simulator's built-in checkpoints use.
        ckpt_stem = folder / f"rl_model_{total_timesteps}_steps"
        model.save(str(ckpt_stem))
        # Record the level so discovery does not have to guess from the name.
        (folder / ".level").write_text(level)
        # And the parameters, so Simulate can rebuild the same envs. C_STM in
        # particular changes the observation space, so a checkpoint loaded
        # without them fails on a shape mismatch.
        model_params.save(folder, applied_params)

        curve_png = _plot_curve(cb.reward_curve, level, folder / "training_curve.png")
        final_reward = cb.reward_curve[-1][1] if cb.reward_curve else float("nan")

        venv.close()

        summary = (
            f"Trained {LEVEL_LABELS.get(level, level)} for {total_timesteps:,} timesteps "
            f"in {elapsed:.0f}s. Final mean episode reward: {final_reward:.3f}.\n\n"
            f"{model_params.summarise(applied_params)}\n\n"
            f"Saved as '{folder_name}'. This is a demo-scale model -- the built-in "
            f"checkpoints are 100-200M timesteps on GPU, so expect this one to read poorly."
        )

        return {
            "model_id": f"session:{folder_name}/{ckpt_stem.name}",
            "model_path": str(ckpt_stem),
            "folder": folder_name,
            "curve_png": curve_png,
            "summary": summary,
            "level": level,
            "model_parameters": applied_params,
            "zip_path": str(ckpt_stem) + ".zip",
        }
    finally:
        os.chdir(prev_cwd)
