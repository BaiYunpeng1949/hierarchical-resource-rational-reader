"""
Runs the reading simulator for the app and renders its figures.

This mirrors what simulator_pipeline.py does from the command line, with two
additions the UI needs: the choice of checkpoint per level, and an agent cache.
Building a ReaderAgent loads three PPO models and takes a few seconds, so agents
are kept warm and reused whenever the same combination of checkpoints comes back.
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import model_params
import model_registry as registry

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = registry.SIM_DIR
VIZ_DIR = SIM_DIR / "results" / "visualize_fixations"
STIM_ASSET_DIR = (
    VIZ_DIR / "assets" / "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400" / "simulate"
)
METADATA_PATH = STIM_ASSET_DIR / "metadata.json"

for p in (str(SIM_DIR), str(VIZ_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore")

from simulator import ReaderAgent, run_batch_simulations, TIME_CONDITIONS
from process_sim_results import build_trials, load_json
from plot_scanpaths import find_image, plot_trial_on_image
from plot_heatmaps import plot_heat_for_trial

# Number of stimuli in the bundled set. Arbitrary text is not supported yet: each
# stimulus carries precomputed per-sentence appraisal scores and a rendered image.
NUM_STIMULI = 9

# Rendered figures go to the system temp directory, not into the repository.
# Gradio will only serve a file from the working directory or from temp, and
# refuses anything else with InvalidPathError -- the figures used to be written
# next to the simulation results, which is neither.
FIGURE_DIR = Path(tempfile.gettempdir()) / "reader_app_figures"

_agent_cache = {}


def _cache_key(model_ids, asset_key):
    # The asset is part of the key: it is read when the sentence env is built, so
    # a cached agent is already bound to whichever one was selected then.
    return tuple(model_ids.get(level) or "" for level in registry.LEVELS) + (asset_key,)


def training_parameters(model_ids):
    """
    The paper parameters recorded for the selected checkpoints, merged.

    A checkpoint trained under a non-default C_STM or n_fov only makes sense in
    an env built the same way, so the values travel with the checkpoint rather
    than being asked of the user again. The three levels own disjoint
    parameters, so merging them cannot conflict.
    """
    merged = {}
    for level in registry.LEVELS:
        m = registry.lookup(model_ids.get(level))
        if not m:
            continue
        merged.update(model_params.normalise(m.get("model_parameters"), level=level))
    return merged


def get_agent(model_ids, asset_key=model_params.DEFAULT_ASSET_KEY):
    """A ReaderAgent for this combination of checkpoints, reused when possible."""
    key = _cache_key(model_ids, asset_key)
    if key in _agent_cache:
        return _agent_cache[key]

    specs = {}
    for level in registry.LEVELS:
        spec = registry.spec_for(model_ids.get(level))
        if spec:
            specs[level] = spec

    prev = os.getcwd()
    os.chdir(SIM_DIR)
    try:
        # ReaderAgent builds all three envs here, and the envs latch the
        # structural constants -- and the sentence observation asset -- in
        # __init__, so the patches only need to span the construction. The cache
        # key is the checkpoint ids plus the asset, which determine these values,
        # so a cached agent already has the right ones.
        with model_params.constants_applied(training_parameters(model_ids)), \
                model_params.observation_asset_applied(asset_key):
            agent = ReaderAgent(model_specs=specs)
    finally:
        os.chdir(prev)

    # Only one combination is kept warm; each agent holds three PPO models and a
    # free Space has limited RAM.
    _agent_cache.clear()
    _agent_cache[key] = agent
    return agent


def describe_selection(model_ids):
    """Human-readable note about which checkpoint each level will use."""
    lines = []
    for level in registry.LEVELS:
        m = registry.lookup(model_ids.get(level))
        label = m["label"] if m else "config default"
        lines.append(f"- {registry.LEVEL_LABELS[level]}: {label}")
    return "\n".join(lines)


def run_simulation(
    model_ids,
    stimulus_ids,
    time_conditions,
    rho=0.29,
    w_skip=0.70,
    coverage=1.30,
    trials=1,
    make_heatmaps=False,
    asset_key=model_params.DEFAULT_ASSET_KEY,
    output_dir=None,
    progress=None,
):
    """
    Simulate and render. Returns (image_paths, summary_markdown).

    stimulus_ids is a list of ints, time_conditions a list like ["30s", "60s"].
    """
    if not stimulus_ids:
        raise ValueError("Pick at least one stimulus.")
    if not time_conditions:
        raise ValueError("Pick at least one time condition.")

    bad = [c for c in time_conditions if c not in TIME_CONDITIONS]
    if bad:
        raise ValueError(f"Unknown time condition(s): {', '.join(bad)}")

    asset = model_params.observation_asset(asset_key)

    if progress:
        progress(0.05, "Loading models")
    agent = get_agent(model_ids, asset_key)

    if progress:
        progress(0.25, "Simulating")

    # Per-episode parameters recorded at training time (kappa, w_reg) are used as
    # the base, so a checkpoint reads under the parameters it was trained under.
    # The sliders below win where they overlap -- the user's explicit choice is
    # never silently discarded.
    trained = training_parameters(model_ids)
    word_params = model_params.reset_params(trained, "word_recognizer")
    sentence_params = model_params.reset_params(trained, "sentence_reader")
    text_params = model_params.reset_params(trained, "text_reader")
    word_params["rho_inflation_percentage"] = rho
    sentence_params["w_skip_degradation_factor"] = w_skip
    text_params["coverage_factor"] = coverage

    prev = os.getcwd()
    os.chdir(SIM_DIR)
    try:
        output = run_batch_simulations(
            simulator=agent,
            stimulus_ids=list(stimulus_ids),
            time_conditions=list(time_conditions),
            word_recognizer_params=word_params,
            sentence_reader_params=sentence_params,
            text_reader_params=text_params,
            num_trials=int(trials),
            output_dir=str(SIM_DIR / "simulated_results" / "_app_tmp"),
        )
        results = output["results"]

        if progress:
            progress(0.7, "Rendering figures")

        out_dir = Path(output_dir) if output_dir else FIGURE_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = load_json(METADATA_PATH)
        trials_data = build_trials(results, meta)

        images = []
        for trial in trials_data:
            stim_idx = trial.get("stimulus_index")
            cond = trial.get("time_constraint")
            img_path = find_image(STIM_ASSET_DIR, stim_idx)
            if img_path is None:
                continue
            pid = trial.get("participant_index", 0)

            scan_path = out_dir / f"scanpath_stim{stim_idx}_{cond}_{pid}.png"
            plot_trial_on_image(
                trial, img_path, scan_path,
                participant="simulation", label="simulation",
                dot_size=90.0, line_width=2.0, alpha_dots=0.4, alpha_lines=0.4,
            )
            images.append(str(scan_path))

            if make_heatmaps:
                heat_path = out_dir / f"heatmap_stim{stim_idx}_{cond}_{pid}.png"
                plot_heat_for_trial(
                    trial, STIM_ASSET_DIR, heat_path,
                    participant="simulation", label="simulation",
                    sigma_px=60.0, alpha_max=0.65, gamma=0.6, cmap_name="RdBu_r",
                )
                images.append(str(heat_path))
    finally:
        os.chdir(prev)

    if progress:
        progress(1.0, "Done")

    n_fix = sum(len(t.get("fixation_data", [])) for t in trials_data)
    summary = (
        f"**{len(trials_data)} trial(s)** across "
        f"{len(stimulus_ids)} stimulus/stimuli x {len(time_conditions)} condition(s), "
        f"{n_fix} fixations total.\n\n"
        f"**Models used**\n{describe_selection(model_ids)}\n\n"
        f"**Parameters** rho={rho}, w_IL={w_skip}, w_RP={coverage}, trials={trials}\n\n"
        f"**Parafoveal preview** {asset.label}"
        + (f" - {asset.note}" if asset.note else "")
        + "\n\n"
        f"{model_params.summarise(trained)}"
    )
    return images, summary
