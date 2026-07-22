"""
A reading simulator.

Takes user inputs, runs simulator.py, analyzes it, and renders a scanpath for each simulated trial.

TODO:
- Determine which parameters can be edited by the user in the simulator
- Determine how parameters are changed (sliders, text boxes, buttons)
- Allow user to input their own text for the simulator
"""
import argparse
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

# Double check pathing
SIM_DIR = Path(os.path.expanduser(
    "~/PycharmProjects/hierarchical-resource-rational-reader/"
    "step5/simulators/simulator_v20250604"
))
VIZ_DIR = SIM_DIR / "results" / "visualize_fixations"
STIM_ASSET_DIR = VIZ_DIR / "assets" / \
    "08_15_09_07_10_images_W1920H1080WS16_LS40_MARGIN400" / "simulate"
METADATA_PATH = STIM_ASSET_DIR / "metadata.json"

sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(VIZ_DIR))
# Silence annoying warning
# double check?
warnings.filterwarnings("ignore")

from simulator import ReaderAgent, run_batch_simulations
from process_sim_results import build_trials, load_json
from plot_scanpaths import find_image, plot_trial_on_image

# Used for 
def parse_int_list(spec):
    spec = spec.strip()
    if "-" in spec:
        a, b = (int(x) for x in spec.split("-"))
        return list(range(a,b+1))
    return [int(x) for x in spec.split(",") if x.strip()]

def run_sim(stim_id, conds, rho, w, cov, trials, kappa=None, w_reg=None):
    """run the model for every stimulus and condition. Returns results list."""
    os.chdir(SIM_DIR)
    sys.path.insert(0, str(SIM_DIR))
    agent = ReaderAgent()

    # Optimized (tunable) parameters from the paper. kappa and w_reg fall back to
    # the model defaults (2.50 and 1.0) when not supplied.
    wr_params = {"rho_inflation_percentage": rho}
    sr_params = {"w_skip_degradation_factor": w}
    if kappa is not None:
        wr_params["kappa"] = kappa
    if w_reg is not None:
        sr_params["w_regression_cost"] = w_reg

    output = run_batch_simulations(
        simulator=agent,
        stimulus_ids=stim_id,
        time_conditions=conds,
        word_recognizer_params=wr_params,
        sentence_reader_params=sr_params,
        text_reader_params={"coverage_factor": cov},
        num_trials=trials,
        output_dir=str(SIM_DIR / "simulated_results" / "_run_simulator_tmp"),
    )
    return output["results"]

def render_scanpaths(results, output_dir, dot_size, line_width, alpha):
    """Conver results to scan paths"""
    sys.path.insert(0, str(VIZ_DIR))
    meta = load_json(METADATA_PATH)
    trials = build_trials(results, meta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []

    for trial in trials:
        stim_idx = trial.get("stimulus_index")
        time_constraints = trial.get("time_constraint")
        img_path = find_image(STIM_ASSET_DIR, stim_idx)
        if img_path is None:
            print(f"No image found for stimulus {stim_idx}")
            continue

        out_path = output_dir / f"scanpath_stim{stim_idx}_{time_constraints}_{trial.get('participant_index',0)}.png"
        plot_trial_on_image(trial, img_path, out_path,
                            participant="simulation", label="simulation",
                            dot_size=dot_size, line_width=line_width, alpha_dots=alpha, alpha_lines=alpha)
        n_fix = len(trial.get("fixation_data", []))
        print(f"{out_path} -> {n_fix} fixations")
        written.append(out_path)
    return written

def main():
    args = argparse.ArgumentParser(description="Run the reading simulator and render a scanpath for each trial.")
    args.add_argument("--stimuli", default="0", help="ids, e.g. '0', '0,3', '0-8'")
    args.add_argument("--conds", default="30s", help="e.g. '30s' or '30s,60s,90s'")
    args.add_argument("--trials", type=int, default=1, help="trials per stim x cond")
    #
    args.add_argument("--rho", type=float, default=0.29, help="word vigor (rho_inflation_percentage)")
    args.add_argument("--w", type=float, default=0.70, help="skip tendency (w_skip_degradation_factor)")
    args.add_argument("--cov", type=float, default=1.30, help="coverage drive (coverage_factor / paper w_RP)")
    args.add_argument("--kappa", type=float, default=None,
                      help="processing time per bit of lexical info gain (paper k, default 2.50)")
    args.add_argument("--w-regression-cost", type=float, default=None, dest="w_regression_cost",
                      help="cost of regressions (paper w_reg, default 1.0)")
    #
    args.add_argument("--out", default="scanpaths_out", help="output directory")
    args.add_argument("--dot-size", type=float, default=90.0)
    args.add_argument("--line-width", type=float, default=2.0)
    args.add_argument("--alpha", type=float, default=0.4, help="dot/line transparency")
    parsed_args = args.parse_args()

    out_dir = Path(parsed_args.out).resolve()

    stimulus_ids = parse_int_list(parsed_args.stimuli)
    conds = [c.strip() for c in parsed_args.conds.split(",") if c.strip()]

    print(f"Simulating stimuli={stimulus_ids} conds={conds} "
          f"rho={parsed_args.rho} w={parsed_args.w} cov={parsed_args.cov} "
          f"kappa={parsed_args.kappa if parsed_args.kappa is not None else '2.50 (default)'} "
          f"w_reg={parsed_args.w_regression_cost if parsed_args.w_regression_cost is not None else '1.0 (default)'}")
    results = run_sim(stimulus_ids, conds, parsed_args.rho, parsed_args.w, parsed_args.cov,
                      parsed_args.trials, kappa=parsed_args.kappa, w_reg=parsed_args.w_regression_cost)
    written = render_scanpaths(results, out_dir, parsed_args.dot_size, parsed_args.line_width, parsed_args.alpha)
    print(f"\nDone. {len(written)} scanpaths in {out_dir}")

if __name__ == "__main__":
    main()
