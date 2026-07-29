"""
Discovery of the PPO checkpoints the app can simulate with.

Three sources feed the model dropdowns:

- built-in   : the checkpoints committed under sub_models/training/saved_models
- trained    : checkpoints produced by the Train tab during this session
- uploaded   : .zip checkpoints the user supplied through the Upload control

Built-in checkpoints live inside the repo. The other two live in a session
directory that is wiped when the process restarts -- on a free Hugging Face
Space the filesystem is ephemeral, so anything a user wants to keep has to be
downloaded before they close the tab.
"""
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import model_params

SIM_DIR = Path(__file__).resolve().parent.parent / "step5" / "simulators" / "simulator_v20250604"
BUILTIN_MODELS_DIR = SIM_DIR / "sub_models" / "training" / "saved_models"

# Which level each checkpoint can be used for. The three envs have different
# observation/action spaces, so a checkpoint is only loadable at its own level.
LEVELS = ("word_recognizer", "sentence_reader", "text_reader")

LEVEL_LABELS = {
    "word_recognizer": "Word recognition",
    "sentence_reader": "Sentence reading",
    "text_reader": "Text reading",
}

# The committed checkpoint folders carry their level in the name. Anything that
# does not match is reported as unknown rather than guessed at.
_LEVEL_PATTERNS = (
    ("word_recognizer", re.compile(r"word[_-]?recognition|word[_-]?activation", re.I)),
    ("sentence_reader", re.compile(r"sentence[_-]?rea", re.I)),
    ("text_reader", re.compile(r"text[_-]?read", re.I)),
)


def _session_root():
    """Session scratch dir for trained/uploaded checkpoints, created on first use."""
    root = Path(os.environ.get("READER_APP_SESSION_DIR", "")) if os.environ.get(
        "READER_APP_SESSION_DIR"
    ) else Path(tempfile.gettempdir()) / "reader_app_session"
    root.mkdir(parents=True, exist_ok=True)
    return root


SESSION_MODELS_DIR = _session_root() / "models"
SESSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def infer_level(name):
    """Best-effort level for a checkpoint folder name; None when it cannot be told."""
    for level, pattern in _LEVEL_PATTERNS:
        if pattern.search(name):
            return level
    return None


def _checkpoints_in(folder):
    """The .zip checkpoints inside one saved-model folder, newest last."""
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.zip"), key=lambda p: p.name)


def discover_models():
    """
    Every checkpoint the app can offer, as a list of dicts:

        {id, label, level, source, model_path, folder, checkpoint}

    `id` is what the dropdowns store; resolve it with `lookup`.
    """
    found = []

    def collect(root, source):
        if not root.is_dir():
            return
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            # An explicit .level marker (written on upload/train) beats name matching.
            level = _declared_level(folder) or infer_level(folder.name)
            # The paper parameters this checkpoint was trained under. Built-in and
            # uploaded checkpoints have none recorded, which means "the defaults".
            params = model_params.load(folder)
            changed = model_params.non_default(params)
            for ckpt in _checkpoints_in(folder):
                # PPO.load wants the path without the .zip suffix.
                stem = ckpt.with_suffix("")
                label = f"[{source}] {folder.name} / {ckpt.stem}"
                if changed:
                    label += " (" + ", ".join(
                        f"{model_params.BY_KEY[k].symbol}={v}"
                        for k, v in sorted(changed.items())
                    ) + ")"
                found.append(
                    {
                        "id": f"{source}:{folder.name}/{ckpt.stem}",
                        "label": label,
                        "level": level,
                        "source": source,
                        "model_path": str(stem),
                        "folder": folder.name,
                        "checkpoint": ckpt.stem,
                        "model_parameters": params,
                    }
                )

    collect(BUILTIN_MODELS_DIR, "built-in")
    collect(SESSION_MODELS_DIR, "session")
    return found


def models_for_level(level):
    """Checkpoints usable at `level`, plus any whose level could not be inferred."""
    out = []
    for m in discover_models():
        if m["level"] == level or m["level"] is None:
            out.append(m)
    return out


def choices_for_level(level):
    """(label, id) pairs for a Gradio dropdown."""
    return [(m["label"], m["id"]) for m in models_for_level(level)]


def lookup(model_id):
    """The model dict for an id from a dropdown, or None."""
    for m in discover_models():
        if m["id"] == model_id:
            return m
    return None


def spec_for(model_id):
    """
    Turn a dropdown id into the model_spec that simulator.ReaderAgent accepts.

    Returns None when the id is empty or unknown, which makes the simulator fall
    back to whatever config.yaml names for that level.
    """
    m = lookup(model_id)
    if m is None:
        return None
    return {"model_path": m["model_path"]}


def register_upload(uploaded_path, level, name=None):
    """
    Copy an uploaded .zip checkpoint into the session store so it shows up in the
    dropdowns. Returns (model_id, message).

    The level is taken from the user rather than inferred -- an uploaded file can
    be named anything, and loading a checkpoint at the wrong level fails deep
    inside PPO with an unhelpful shape error.
    """
    if not uploaded_path:
        return None, "No file supplied."
    src = Path(uploaded_path)
    if src.suffix != ".zip":
        return None, f"Expected a .zip checkpoint, got '{src.suffix or 'no extension'}'."
    if level not in LEVELS:
        return None, f"Pick which level this checkpoint is for (one of {', '.join(LEVELS)})."

    stamp = time.strftime("%m%d_%H%M%S")
    folder_name = name.strip() if name and name.strip() else f"uploaded_{level}_{stamp}"
    folder = SESSION_MODELS_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / src.name
    shutil.copyfile(src, dest)

    # Record the level explicitly, since the folder name may not reveal it.
    (folder / ".level").write_text(level)

    model_id = f"session:{folder_name}/{dest.stem}"
    return model_id, f"Uploaded '{src.name}' as {folder_name}, available for {LEVEL_LABELS[level]}."


def _declared_level(folder):
    marker = folder / ".level"
    if marker.is_file():
        text = marker.read_text().strip()
        if text in LEVELS:
            return text
    return None
