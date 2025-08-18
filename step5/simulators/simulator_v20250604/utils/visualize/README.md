# Visualize Scanpaths

Utilities to (1) convert simulation logs into **image‑aligned fixation coordinates**, and (2) **plot** scanpaths for both human and simulation data with color‑coded saccades (forward / skip / regression).

---

## Folder structure

```
visualize_scanpaths/
├─ assets/
│  ├─ <images_dir>/                      # stimulus images, filenames: 0.png, 1.png, ... (jpg/jpeg/webp also OK)
│  ├─ metadata.json                      # stimulus metadata used to map word/letter → pixel coords
│  ├─ 11_18_17_40_integrated_corrected_human_scanpath.json  # human scanpaths (list of trials)
│  └─ simulation_scanpaths.json          # simulation scanpaths (list of trials) — produced by process_sim_results.py
├─ vis/
│  ├─ human/                             # rendered human plots (created automatically)
│  └─ simulation/                        # rendered simulation plots (created automatically)
├─ process_sim_results.py                # convert sim logs → image-aligned scanpath JSON
├─ plot_scanpaths.py                     # plot scanpaths over images; colors forward/skip/regression
└─ README.md
```

---

## What each file does

### `process_sim_results.py`
Converts simulation output (with sentence/word/letter indices) into a **plot‑ready JSON** with fixation coordinates on the stimulus image.

- **Inputs**:  
  - `all_simulation_results.json` (your raw sim output)  
  - `metadata.json` (stimulus words + letters + image size)

- **Output**: `simulation_scanpaths.json` (a list of trials). Each trial has:
  ```json
  {
    "stimulus_index": 0,
    "participant": "simulation",
    "time_constraint": 30,
    "fixation_data": [
      {
        "fix_x": 650.5,
        "fix_y": 382.0,
        "norm_fix_x": 0.339,
        "norm_fix_y": 0.354,
        "fix_duration": 270.0,           // milliseconds
        "word_index": 42,                 // global word index in text
        "recognized_word": "example",     // first recognized (may be null)
        "recognized_words": ["example"]   // full list (may be empty)
      }
      // ...
    ]
  }
  ```

- **Coordinate rule** (per fixation):  
  1) If **exactly one** sampled letter → use **that letter center**.  
  2) If **multiple** letters → average their centers.  
  3) If none/invalid → fall back to **word bbox center**.


### `plot_scanpaths.py`
Plots scanpaths on top of the stimulus image and writes **one PNG per trial**.

- **Coloring rules** (computed from `word_index` per fixation):
  - **Red line** = normal forward/adjacent/refixation
  - **Blue line** = **skip** (forward jump > 1 word)
  - **Green line** = **regression** (`next_word_index < furthest_word_seen_so_far`)
  - **Green dot** = destination fixation of a regressive saccade  
    (the first fixation is always red)
- **Transparency / size** are configurable so text remains readable.
- **Output directories** are split automatically for human vs simulation and files are **overwritten** if they already exist.

---

## Data origins

- **`metadata.json`** should come from your stimulus builder. It must include, for each stimulus:
  - image size (`config["img size"]`),
  - per‑word entries with `"word_bbox"`,
  - per‑letter entries with `"letters metadata" → "letter boxes"` (left/right/top/bottom).
- **Human scanpaths**: a JSON list of trials matching the schema above (`participant: "human"` is preferred).  
- **Images**: place under `assets/<images_dir>/` with filenames `0.png, 1.png, ...` (or `.jpg/.jpeg/.webp`).

> If you need to regenerate `metadata.json`, use your stimulus generation pipeline (not covered here). The plotting script only reads it indirectly via the already processed simulation JSON.

---

## Setup

- **Python**: 3.9+
- **Packages**:
  ```bash
  pip install pillow matplotlib
  ```
  (`process_sim_results.py` uses only the standard library; `plot_scanpaths.py` needs Pillow + Matplotlib.)

---

## How to replicate

### 1) Convert simulation logs → plot‑ready JSON
```bash
python process_sim_results.py \
  --simulation 20250814_1555_trials1_stims9_conds3 \
  --out_json   assets/simulation_scanpaths.json
```
- Inputs only the folder name is enough. Folders are in `/step5/simulators/simulator_v20250604/simulated_results/`
- Produces `assets/simulation_scanpaths.json` (list of trials).

### 2) Plot scanpaths

**Simulation only** (to a dedicated folder):
```bash
python plot_scanpaths.py \
  --out_root scanpaths \
  --sim_out_dir scanpaths/simulation \
  --default_participant simulation \
  assets/simulation_scanpaths.json
```

**Human only** (separate destination):
```bash
python plot_scanpaths.py \
  --out_root scanpaths \
  --human_out_dir scanpaths/human \
  --default_participant human \
  assets/11_18_17_40_integrated_corrected_human_scanpath.json
```

**Both**, auto‑split under `vis/`:
```bash
python plot_scanpaths.py \
  --out_root scanpaths \
  assets/11_18_17_40_integrated_corrected_human_scanpath.json \
  assets/simulation_scanpaths.json
```

**Output names** look like:
```
scanpaths/simulation/stim0_simulation_time30s.png
scanpaths/human/stim0_human_time30s.png
```

### CLI options (plotting)
- `--dot_size` (default `90.0`) — fixation dot size  
- `--line_width` (default `2.0`) — saccade line width  
- `--alpha_dots` (default `0.5`) — dot transparency
- `--alpha_lines` (default `0.5`) — line transparency
- `--sim_out_dir` / `--human_out_dir` — custom output dirs (defaults to `out_root/simulation` and `out_root/human`)
- `--default_participant` — fallback label if a trial misses `participant`

---

## Notes & troubleshooting

- **Coloring depends on `word_index`** per fixation. If your human JSON lacks it or uses another name, add the field or tell the plotting script how to find it. (It already tries `word_index` or `global_word_index`.)
- **Regressions** are detected against the **furthest‑so‑far** word index. If you want *first‑pass‑only* skip logic, we can add a flag to restrict skip detection to unseen words.
- If plots look too opaque, lower `--alpha_dots` and `--alpha_lines` (e.g., `0.4`).
- If you have higher‑res images, you may want larger `--dot_size` (e.g., `120`).

---

## License
Internal research tooling. Adapt as needed for your pipeline.
---

## Heatmap plotting (`plot_heatmaps.py`)

Create **duration‑weighted gaze heatmaps** over each stimulus image.
Each fixation contributes a Gaussian kernel centered at `(fix_x, fix_y)` with weight = `fix_duration` (milliseconds).
If `fix_duration` is missing, weight defaults to `1`.

### Inputs
- One or more JSON files (each a **list of trials** with `fixation_data`).
- A directory of stimulus images (`--images_dir`). The script matches `0.png`, `image_0.png`, `stim0.jpg`, etc.

### Outputs
- PNGs written to `heatmap_plots/human/` and `heatmap_plots/simulation/` (or custom dirs).
- Filenames: `stim{index}_{label}_time{tc}.png` (e.g., `stim7_human-23_time90.png`, `stim0_simulation-0_time30.png`).

### Global vs per‑image color scaling
By default each image is normalized to its **own** maximum ("per‑image"), which makes
30s and 90s trials look similarly saturated.
To compare images fairly across **different time budgets**, use **fixed** normalization.

- **Fixed 90 s ceiling** (recommended for parallel comparisons):
  ```bash
  python plot_heatmaps.py     --images_dir assets/<images_dir>     --out_root heatmap_plots     --norm_mode fixed --vmax_ms 90000     assets/simulation_scanpaths.json
  ```
  Any pixel with ≥90,000 ms accumulated dwell will hit full saturation; lower values scale proportionally.

- If 90 s is too high/low, pick another unified ceiling, e.g. **60 s** or **45 s**:
  ```bash
  --norm_mode fixed --vmax_ms 60000   # 60 s
  --norm_mode fixed --vmax_ms 45000   # 45 s
  ```

### Examples

**Only simulation trials**:
```bash
python plot_heatmaps.py   --out_root heatmap_plots   --only simulation   --norm_mode fixed --vmax_ms 3000   --show_totals   --norm_to_tc   assets/simulation_scanpaths.json
```

**Only human trials**:
```bash
python plot_heatmaps.py   --out_root heatmap_plots   --only human   --norm_mode fixed --vmax_ms 3000   --show_totals   --norm_to_tc   assets/11_18_17_40_integrated_corrected_human_scanpath.json
```

**Both (auto‑split to subfolders)**:
```bash
python plot_heatmaps.py   --out_root heatmap_plots   --norm_mode per-image   assets/11_18_17_40_integrated_corrected_human_scanpath.json   assets/simulation_scanpaths.json
```

### Useful flags
- `--sigma_px` (default **60**) — Gaussian spread per fixation in pixels (larger → smoother heatmap)
- `--alpha_max` (default **0.65**) — maximum overlay opacity
- `--gamma` (default **0.6**) — opacity curve; lower boosts mid‑range intensity
- `--cmap` (default **RdBu_r**) — colormap (`RdBu_r` = blue high / red low)
- `--only {human|simulation}` / `--exclude {human|simulation}` — filter which trials to render
- `--verbose` — print which image file was matched; helpful if names vary

