"""
The paper's tunable model parameters, as knobs the Train tab can set.

These are the "Assumptions and tunable parameters in the reading model" of
Extended Data Table 1 -- cognitive constants, not RL hyperparameters. Learning
rate changes how well the policy is fitted; these change what the policy is
being fitted *to*, so a checkpoint is only meaningful alongside the values it was
trained under. Every trained run therefore writes its values next to the
checkpoint (see `save`/`load`), and the Simulate tab reads them back.

Three ways a parameter reaches an env, recorded per parameter as `apply`:

  "constant"  -- a module-level constant the env latches in __init__
                 (n_fov, C_STM). Patched on the module for the duration of env
                 construction and restored afterwards, the same trick
                 trainer._training_mode uses for rl.mode.
  "reset"     -- passed through env.reset(params=...) each episode
                 (kappa, rho, w_reg). SB3 resets with no arguments, so training
                 injects these with a wrapper.
  "offline"   -- baked into precomputed assets, not read at run time
                 (n_parafov). Not a slider: it is chosen by swapping in a
                 differently generated asset file, see OBSERVATION_ASSETS.

Defaults here are the values the code uses today, so leaving the panel alone
reproduces current training exactly. Where the paper's fitted value differs it is
recorded in `paper_value` and shown in the UI, not silently applied.
"""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent / "step5" / "simulators" / "simulator_v20250604"

# The env packages import each other with bare names, as they do under trainer.py.
for _p in (str(SIM_DIR), str(SIM_DIR / "sub_models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PARAMS_FILENAME = "model_params.json"


class Param:
    """One row of Extended Data Table 1, plus how the code reaches it."""

    def __init__(self, key, symbol, label, level, apply, default, paper_value,
                 low, high, step, method, explanation, target=None, note=None):
        self.key = key
        self.symbol = symbol
        self.label = label
        self.level = level            # which of the three agents it affects
        self.apply = apply            # "constant" | "reset" | "offline"
        self.default = default        # what the code uses today
        self.paper_value = paper_value
        self.low = low
        self.high = high
        self.step = step
        self.method = method          # "L" (literature) or "O" (optimization)
        self.explanation = explanation
        self.target = target          # ("module path", "ATTR") for apply="constant"
                                      # or the env's reset key for apply="reset"
        self.note = note

    @property
    def is_integer(self):
        return isinstance(self.default, int)

    def coerce(self, value):
        """Cast and range-check one user-supplied value."""
        value = int(round(float(value))) if self.is_integer else float(value)
        if not (self.low <= value <= self.high):
            raise ValueError(
                f"{self.symbol} = {value} is outside the paper's range "
                f"[{self.low}, {self.high}]."
            )
        return value

    def describe(self):
        paper = (
            f"paper {self.paper_value}"
            if self.paper_value == self.default
            else f"paper {self.paper_value}, code default {self.default}"
        )
        # `method` (literature vs optimization) is recorded on the Param but left
        # out of the UI text.
        return f"{self.explanation}. Range [{self.low}, {self.high}]; {paper}."


# Ordered as the paper's table. `target` for "constant" params names the module
# attribute to patch; for "reset" params it is the key the env's reset() reads.
PARAMS = [
    Param(
        key="n_fov", symbol="n_fov", label="n_fov - foveal span",
        level="word_recognizer", apply="constant",
        default=8, paper_value=8, low=7, high=9, step=1, method="L",
        explanation="Number of high-acuity letters visible in foveal vision",
        target=("sub_models.word_recognition_v0807.Constants", "FOVEAL_SIZE"),
    ),
    Param(
        key="n_parafov", symbol="n_parafov", label="n_parafov - parafoveal preview",
        level="word_recognizer", apply="offline",
        default=2, paper_value=2, low=1, high=3, step=1, method="L",
        explanation="Number of previewable letters in the next word (parafoveal vision)",
        note=(
            "Applied when the sentence assets are generated, not at run time: it "
            "sets how many clear letters the BERT predictability estimates assume. "
            "Choose it on the Simulate tab, which swaps in an asset generated at "
            "that value; see OBSERVATION_ASSETS."
        ),
    ),
    Param(
        key="kappa", symbol="kappa", label="kappa - time cost per bit",
        level="word_recognizer", apply="reset",
        default=2.50, paper_value=2.50, low=2.0, high=4.0, step=0.05, method="O",
        explanation="Processing time cost per bit of lexical information gain, ms/bit",
        target="kappa",
    ),
    Param(
        key="rho", symbol="rho", label="rho - non-fixation overhead",
        level="word_recognizer", apply="reset",
        default=0.20, paper_value=0.29, low=0.1, high=0.3, step=0.01, method="O",
        explanation="Non-fixation overhead fraction",
        target="rho_inflation_percentage",
    ),
    Param(
        key="c_stm", symbol="C_STM", label="C_STM - short-term memory capacity",
        level="word_recognizer", apply="constant",
        default=5, paper_value=5, low=5, high=9, step=1, method="L",
        explanation="Capacity of the short-term memory buffer (parallel word candidates)",
        target=("sub_models.word_recognition_v0807.Constants", "WORKING_MEMORY_SIZE"),
        note=(
            "Changes the size of the belief distribution, so it changes the word "
            "recogniser's observation space. A checkpoint trained at one C_STM "
            "cannot be loaded at another."
        ),
    ),
    Param(
        key="w_reg", symbol="w_reg", label="w_reg - regression cost",
        level="sentence_reader", apply="reset",
        default=1.00, paper_value=0.80, low=0.0, high=1.0, step=0.01, method="O",
        explanation="Weighting the cost of regressions",
        target="w_regression_cost",
    ),
]

BY_KEY = {p.key: p for p in PARAMS}

# Table rows that the Train tab deliberately does not offer, and why. Shown in
# the UI so the panel reads as the full table rather than an arbitrary subset.
NOT_TUNABLE_AT_RUNTIME = {
    "d_saccade": "Fixed at 25 ms in TransitionFunction.calc_individual_saccade_duration_ms.",
    "w_IL": (
        "Sampled uniformly from [0.5, 1.0] during training on purpose -- the "
        "sentence policy is amortised over it and reads it from its observation. "
        "Set it on the Simulate tab instead (w_skip)."
    ),
    "w_RP": (
        "Sampled during training for the same reason. Set it on the Simulate tab "
        "instead (coverage)."
    ),
    "tau_LTM / tau_H / tau_L": (
        "Used by the offline Kintsch comprehension runner, not by the three RL envs."
    ),
}


# ---------------------------------------------------------------- n_parafov assets
#
# n_parafov is not a run-time knob: it decides how many letters of the next word
# BERT is allowed to see when the sentence observation asset is generated, so
# picking a value means picking a file. The choices below are those files.
#
# All four are committed data. The tooling that produced them is not in the
# repository and the app never regenerates them, so a missing file is restored
# from version control rather than rebuilt.
#
# Two things about them are worth knowing before reading any result:
#
#  1. The shipped asset ("paper") was generated with the RNG in the generator's
#     noisy parafoveal matching left unseeded, so nobody can regenerate it --
#     rebuilding at its own n_parafov=2 disagrees with it on most words. It is
#     kept as the default because it is what the published numbers came from.
#  2. The n=1/2/3 variants share one seed (ASSET_SEED), so they differ from each
#     other only in the preview. Comparisons belong within that set. "paper"
#     versus a variant mixes the preview change with that unseeded draw, and is
#     not a measurement of n_parafov.

SENTENCE_ASSET_DIR = SIM_DIR / "sub_models" / "sentence_read_v0604" / "assets"
VARIANT_SUBDIR = "variants"

# The seed the variants were generated at, recorded because it is part of their
# filenames and their .meta.json sidecars.
ASSET_SEED = 0

PAPER_ASSET = "processed_my_stimulus_with_observations.json"


class ObservationAsset:
    """One selectable sentence observation asset."""

    def __init__(self, key, label, n_parafov, filename, note):
        self.key = key
        self.label = label
        self.n_parafov = n_parafov
        self.filename = filename          # relative to the assets/ directory
        self.note = note

    @property
    def path(self):
        return SENTENCE_ASSET_DIR / self.filename

    @property
    def exists(self):
        return self.path.is_file()



def _variant(n):
    return ObservationAsset(
        key=f"n{n}",
        label=f"n_parafov = {n}" + ("  (rebuild)" if n == 2 else ""),
        n_parafov=n,
        filename=f"{VARIANT_SUBDIR}/processed_my_stimulus_with_observations_n{n}_seed{ASSET_SEED}.json",
        note=f"Built at n_parafov={n}, seed {ASSET_SEED}.",
    )


OBSERVATION_ASSETS = [
    ObservationAsset(
        key="paper",
        label="Paper demo (n_parafov = 2)",
        n_parafov=2,
        filename=PAPER_ASSET,
        note="",
    ),
    _variant(1),
    _variant(2),
    _variant(3),
]

ASSETS_BY_KEY = {a.key: a for a in OBSERVATION_ASSETS}
DEFAULT_ASSET_KEY = "paper"


def observation_asset(key):
    """The selected asset, defaulting to the paper one for an unknown key."""
    return ASSETS_BY_KEY.get(key or DEFAULT_ASSET_KEY, ASSETS_BY_KEY[DEFAULT_ASSET_KEY])


@contextmanager
def observation_asset_applied(key):
    """
    Point the sentence dataset at the selected asset for the duration.

    SentencesManager reads Constants.OBSERVATION_ASSET when the env is built, so
    like constants_applied this only has to span env construction -- but it is
    restored afterwards so a later run in the same process cannot inherit it.
    """
    import importlib

    asset = observation_asset(key)
    if not asset.exists:
        # Every asset is committed data. None of them can be rebuilt from the
        # repository, so a missing one is restored, not regenerated.
        raise FileNotFoundError(
            f"The asset for '{asset.label}' is missing ({asset.path}). "
            "It ships with the repository and cannot be regenerated -- restore "
            "it from version control."
        )

    constants = importlib.import_module("sub_models.sentence_read_v0604.Constants")
    original = constants.OBSERVATION_ASSET
    constants.OBSERVATION_ASSET = asset.filename
    try:
        yield asset
    finally:
        constants.OBSERVATION_ASSET = original


def params_for_level(level):
    """The parameters that actually reach `level`'s env, in table order."""
    return [p for p in PARAMS if p.level == level]


def defaults():
    return {p.key: p.default for p in PARAMS}


def normalise(values, level=None):
    """
    Validate a {key: value} dict and drop anything that does not apply.

    Unknown keys and parameters belonging to another level are ignored rather
    than rejected, so the UI can hand over the whole panel regardless of which
    level is being trained. Out-of-range values raise.
    """
    out = {}
    for key, value in (values or {}).items():
        param = BY_KEY.get(key)
        if param is None or param.apply == "offline":
            continue
        if level is not None and param.level != level:
            continue
        out[key] = param.coerce(value)
    return out


def non_default(values):
    """The subset of `values` that differs from the code default."""
    return {k: v for k, v in (values or {}).items()
            if k in BY_KEY and v != BY_KEY[k].default}


def reset_params(values, level):
    """
    The dict to hand to env.reset(params=...) for `level`.

    Empty when nothing applies, in which case the caller should pass None and let
    the env keep its own defaults.
    """
    out = {}
    for key, value in (values or {}).items():
        param = BY_KEY.get(key)
        if param and param.apply == "reset" and param.level == level:
            out[param.target] = value
    return out


@contextmanager
def constants_applied(values, level=None):
    """
    Patch the module-level constants named by the "constant" parameters.

    The envs read these once, in __init__, so this only has to be held open
    around env construction -- but it is a context manager rather than a setter
    because leaving a patched constant behind would silently contaminate every
    later run in the same process. Restores on error.

    Callers holding trainer._MODE_LOCK are already serialised; this is not
    re-entrant across threads on its own.
    """
    import importlib

    patched = []
    try:
        for key, value in (values or {}).items():
            param = BY_KEY.get(key)
            if not param or param.apply != "constant":
                continue
            if level is not None and param.level != level:
                continue
            module_path, attr = param.target
            module = importlib.import_module(module_path)
            patched.append((module, attr, getattr(module, attr)))
            setattr(module, attr, value)
        yield
    finally:
        for module, attr, original in reversed(patched):
            setattr(module, attr, original)


def save(folder, values):
    """Record the values a checkpoint was trained under, next to the checkpoint."""
    path = Path(folder) / PARAMS_FILENAME
    path.write_text(json.dumps(values, indent=2, sort_keys=True))
    return path


def load(folder):
    """
    The values a checkpoint was trained under, or {} when it predates this or was
    uploaded without them.

    An unreadable file is treated as absent: it only costs the caller the
    non-default parameters, and failing a simulation over it would be worse.
    """
    path = Path(folder) / PARAMS_FILENAME
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def summarise(values):
    """One markdown line per non-default parameter, or a note that none were set."""
    changed = non_default(values)
    if not changed:
        return "Model parameters: all at their defaults."
    rows = ", ".join(
        f"{BY_KEY[k].symbol}={v} (default {BY_KEY[k].default})"
        for k, v in sorted(changed.items())
    )
    return f"Model parameters changed from default: {rows}."
