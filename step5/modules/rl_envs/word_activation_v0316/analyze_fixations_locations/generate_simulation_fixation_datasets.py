import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Default: the baseline model with a pesudo dataset.
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0316_word_activation_v0316_00_baseline/rl_model_10000000_steps/10ep/logs.json"

# Variation 1: noisy oculomotor controller
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0316_word_activation_v0316_01_noisy_oculomotor/rl_model_10000000_steps/10000ep/logs.json"

# Variation 2: noisy action and observation about the word length
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0317_word_activation_v0316_02_laggy_action_and_observation/rl_model_10000000_steps/10000ep/logs.json"

# Variation 3: noisy action and observation + noisy oculomotor control
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_02_laggy_action_and_observation_and_noisy_oculomotor/rl_model_20000000_steps/10000ep/logs.json"

# Variation 4: noisy stuff + five actions
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_04_laggy_action_and_observation_and_noisy_oculomotor_five_actions/rl_model_10000000_steps/10000ep/logs.json"

# Variation 4.1: noise stuff + five fine-tuned action
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_05_laggy_action_and_observation_and_noisy_oculomotor_five_actions/rl_model_10000000_steps/10000ep/logs.json"

# Variation 4.2: noise stuff + five fine-tuned action, 5 noisy oculomotor control
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_06_laggy_action_and_observation_and_noisy_oculomotor_5_five_actions/rl_model_10000000_steps/10000ep/logs.json"

# Variation 4.3: noisy action + five actions, 5 noisy oculo, adaptive region window size
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_06_laggy_action_and_observation_and_noisy_oculomotor_5_five_actions_adaptive_region_window_size/rl_model_10000000_steps/10000ep/logs.json"
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_06_laggy_action_and_observation_and_noisy_oculomotor_5_five_actions_adaptive_region_window_size/rl_model_100000000_steps/10000ep/logs.json"

# Variation 4.4: noisy action + five actions, adaptive noisy oculo, adaptive region window size
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0319_word_activation_v0316_07_laggy_action_and_adaptive_noisy_oculomotor_5_actions_adaptive_region_window_size/rl_model_20000000_steps/10000ep/logs.json"
INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0319_word_activation_v0316_09_laggy_action_and_adaptive_noisy_oculomotor_5_actions_adaptive_region_window_size/rl_model_50000000_steps/10000ep/logs.json"

# Variation 4.5: noisy action + five actions, 3 noisy oculo, adaptive region window size
# INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0319_word_activation_v0316_08_laggy_action_and_3_noisy_oculomotor_5_actions_adaptive_region_window_size/rl_model_20000000_steps/10000ep/logs.json"



OUT_DIR = Path("data/simulation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_INITIAL_FIXATIONS = OUT_DIR / "sim_initial_fixations.csv"
OUT_FORWARD_MULTIPLE = OUT_DIR / "sim_forward_fixations_multiple.csv"
OUT_FORWARD_SINGLE = OUT_DIR / "sim_forward_fixations_single_only.csv"
OUT_REG_INTRAWORD = OUT_DIR / "sim_intraword_regressions_only.csv"
OUT_FIRST_ACTIONS = OUT_DIR / "sim_first_fixation_actions.csv"


# Optional human-data preprocessing
INPUT_MCCONKIE = "mcconkie.csv"
OUT_MCCONKIE_REINDEXED = "mcconkie_reindexed_drop0.csv"


def load_logs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fixation_actions_from_episode(ep: dict):
    actions = []
    for fx in ep.get("fixations", []):
        if fx.get("done", False):
            continue

        executed_action = fx.get("executed_action")
        if executed_action is None:
            continue

        actions.append(int(executed_action))
    return actions


def first_landed_action_from_episode(ep: dict):
    """
    Return the first ACTUAL landed fixation position in the word,
    after oculomotor noise. This is the simulation analogue of
    the first fixation location used in McConkie Figure 1.
    """
    actions = fixation_actions_from_episode(ep)
    return actions[0] if actions else None


def is_multiple_forward_fixations_only(actions):
    """
    Return True if the word received more than one fixation
    and all successive fixations are non-leftward.

    This excludes intraword regressions.
    """
    if len(actions) <= 1:
        return False

    for prev_a, curr_a in zip(actions[:-1], actions[1:]):
        if curr_a < prev_a:
            return False

    return True


def first_intended_action_from_episode(ep: dict):
    """
    Return the agent's first coarse selected action in the episode
    (before oculomotor landing noise).

    Action meanings:
        0 = beginning
        1 = mid_left
        2 = mid_right
        3 = ending
        4 = stop
    """
    for fx in ep.get("fixations", []):
        if fx.get("done", False):
            continue

        intended_action = fx.get("intended_action")
        if intended_action is None:
            intended_action = fx.get("action")

        if intended_action is None:
            continue

        return int(intended_action)

    return None


def count_positions(actions, counter, word_len):
    for a in actions:
        counter[(word_len, a)] += 1


def counts_to_proportion_df(counter, value_name, multiply_by_100=False):
    totals_by_length = defaultdict(int)

    for (word_length, letter_number), count in counter.items():
        totals_by_length[word_length] += count

    rows = []
    for word_length in sorted(totals_by_length.keys()):
        total = totals_by_length[word_length]

        for letter_number in range(word_length):
            count = counter.get((word_length, letter_number), 0)
            value = count / total if total > 0 else 0.0
            if multiply_by_100:
                value *= 100.0

            rows.append(
                {
                    "word_length": word_length,
                    "letter_number": letter_number,
                    value_name: value,
                }
            )

    return pd.DataFrame(rows)


def action_counts_to_proportion_df(counter):
    totals_by_length = defaultdict(int)
    all_actions = [0, 1, 2, 3, 4]

    for (word_length, action), count in counter.items():
        totals_by_length[word_length] += count

    rows = []
    for word_length in sorted(totals_by_length.keys()):
        total = totals_by_length[word_length]

        for action in all_actions:
            count = counter.get((word_length, action), 0)
            proportion = count / total if total > 0 else 0.0

            rows.append(
                {
                    "word_length": word_length,
                    "action": action,
                    "proportion_of_action": proportion,
                }
            )

    return pd.DataFrame(rows)


def preprocess_mcconkie_csv(input_csv: str, output_csv: str):
    """
    Transform McConkie-style indexing:
      - drop letter_number == 0 (space before the word)
      - renormalize the remaining values within each word length
      - shift indexing so original letter 1 becomes 0

    Output stays on a percentage scale summing to 100 within each word length,
    matching the original human dataset convention.
    """
    df = pd.read_csv(input_csv)
    df = df[df["letter_number"] > 0].copy()
    df["letter_number"] = df["letter_number"] - 1

    df["proportion_of_fixation"] = (
        df.groupby("word_length")["proportion_of_fixation"]
        .transform(lambda s: s / s.sum() * 100.0)
    )

    df.to_csv(output_csv, index=False)
    return df


def main():
    logs = load_logs(INPUT_JSON)

    initial_fixation_counts = defaultdict(int)
    forward_multiple_counts = defaultdict(int)
    forward_single_counts = defaultdict(int)
    intraword_regression_counts = defaultdict(int)
    first_action_counts = defaultdict(int)

    for ep in logs:
        word_len = int(ep["word_len"])
        actions = fixation_actions_from_episode(ep)
        first_landed_action = first_landed_action_from_episode(ep)
        first_action = first_intended_action_from_episode(ep)

        if first_landed_action is not None:
            initial_fixation_counts[(word_len, first_landed_action)] += 1

        if first_action is not None:
            first_action_counts[(word_len, first_action)] += 1

        if not actions:
            continue

        if len(actions) == 1:
            count_positions(actions, forward_single_counts, word_len)
        elif is_multiple_forward_fixations_only(actions):
            count_positions(actions, forward_multiple_counts, word_len)

        for prev_a, curr_a in zip(actions[:-1], actions[1:]):
            if curr_a < prev_a:
                intraword_regression_counts[(word_len, curr_a)] += 1

    df_initial_fixations = counts_to_proportion_df(
        initial_fixation_counts,
        "proportion_of_fixation",
        multiply_by_100=False,
    )
    df_forward_multiple = counts_to_proportion_df(
        forward_multiple_counts, "proportion_of_fixation"
    )
    df_forward_single = counts_to_proportion_df(
        forward_single_counts, "proportion_of_fixation"
    )
    df_intraword_reg = counts_to_proportion_df(
        intraword_regression_counts, "probability_of_regression"
    )
    df_first_actions = action_counts_to_proportion_df(first_action_counts)

    df_initial_fixations.to_csv(OUT_INITIAL_FIXATIONS, index=False)
    df_forward_multiple.to_csv(OUT_FORWARD_MULTIPLE, index=False)
    df_forward_single.to_csv(OUT_FORWARD_SINGLE, index=False)
    df_intraword_reg.to_csv(OUT_REG_INTRAWORD, index=False)
    df_first_actions.to_csv(OUT_FIRST_ACTIONS, index=False)

    print("Saved simulation datasets:")
    print(" -", OUT_INITIAL_FIXATIONS)
    print(" -", OUT_FORWARD_MULTIPLE)
    print(" -", OUT_FORWARD_SINGLE)
    print(" -", OUT_REG_INTRAWORD)
    print(" -", OUT_FIRST_ACTIONS)

    if Path(INPUT_MCCONKIE).exists():
        preprocess_mcconkie_csv(INPUT_MCCONKIE, OUT_MCCONKIE_REINDEXED)
        print("Saved transformed human dataset:")
        print(" -", OUT_MCCONKIE_REINDEXED)
    else:
        print(f"Skipped McConkie preprocessing because {INPUT_MCCONKIE} was not found.")


if __name__ == "__main__":
    main()
