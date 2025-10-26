#!/usr/bin/env python3
"""
Trim metadata and add sequential 'word_index_in_text' per image.
- Removes large/heavy keys (same as the original script).
- Adds 'word_index_in_text' = 0..N-1 to each entry in 'words metadata' (or 'words_metadata').

Usage:
  python trim_metadata_with_index.py \
      --in metadata.json \
      --out lightweight_metadata.json
"""

import json
import argparse
from pathlib import Path

# Keys to remove anywhere in the structure
DROP_KEYS = {
    "normalised_foveal_patch",
    "normalised_masked_downsampled_peripheral_view",
    "selected words",
    "selected words indexes",
    "selected words norm indexes",
    "letters metadata",
    "index",
    "normalized index",
    "relative_bbox_foveal_patch",
}

def purge_keys(obj):
    """Recursively remove unwanted keys from dicts/lists."""
    if isinstance(obj, dict):
        return {k: purge_keys(v) for k, v in obj.items() if k not in DROP_KEYS}
    elif isinstance(obj, list):
        return [purge_keys(v) for v in obj]
    else:
        return obj

def add_sequential_word_indices(data: dict) -> dict:
    """
    For each image, find the words list and assign a running 'word_index_in_text'
    from 0..N-1 in reading order.
    Supports either 'words metadata' or 'words_metadata' list names.
    """
    images = data.get("images", [])
    for img in images:
        # Prefer 'words metadata', fall back to 'words_metadata'
        words = img.get("words metadata")
        if words is None:
            words = img.get("words_metadata")

        if isinstance(words, list):
            for i, w in enumerate(words):
                # Do not overwrite if already present
                if isinstance(w, dict) and "word_index_in_text" not in w:
                    w["word_index_in_text"] = int(i)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_file", type=Path, default=Path("metadata.json"))
    ap.add_argument("--out", dest="output_file", type=Path, default=Path("lightweight_metadata.json"))
    args = ap.parse_args()

    print(f"Loading {args.input_file} ...")
    data = json.loads(args.input_file.read_text(encoding="utf-8"))

    print("Filtering out large patches/keys ...")
    cleaned = purge_keys(data)

    print("Adding word_index_in_text ...")
    cleaned = add_sequential_word_indices(cleaned)

    print(f"Writing cleaned JSON to {args.output_file} ...")
    args.output_file.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Done!")

if __name__ == "__main__":
    main()
