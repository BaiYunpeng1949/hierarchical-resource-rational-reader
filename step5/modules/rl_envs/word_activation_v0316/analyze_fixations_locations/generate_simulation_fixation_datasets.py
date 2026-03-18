import json
from collections import defaultdict
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
INPUT_JSON = "/home/baiy4/reader-agent-zuco/step5/data/sim_results/word_activation/0318_word_activation_v0316_06_laggy_action_and_observation_and_noisy_oculomotor_5_five_actions/rl_model_10000000_steps/10000ep/logs.json"


OUT_FORWARD_MULTIPLE = "data/simulation/sim_forward_fixations_multiple.csv"
OUT_FORWARD_SINGLE = "data/simulation/sim_forward_fixations_single_only.csv"
OUT_REG_INTRAWORD = "data/simulation/sim_intraword_regressions_only.csv"


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


def count_positions(actions, counter, word_len):
    for a in actions:
        counter[(word_len, a)] += 1


# def counts_to_proportion_df(counter, value_name):
#     totals_by_length = defaultdict(int)
#     for (word_length, letter_number), count in counter.items():
#         totals_by_length[word_length] += count

#     rows = []
#     for (word_length, letter_number), count in sorted(counter.items()):
#         total = totals_by_length[word_length]
#         value = count / total if total > 0 else 0.0
#         rows.append({
#             "word_length": word_length,
#             "letter_number": letter_number,
#             value_name: value
#         })

#     return pd.DataFrame(rows)


def counts_to_proportion_df(counter, value_name):
    totals_by_length = defaultdict(int)
    max_letter_by_length = defaultdict(int)

    # compute totals and max letter index per word length
    for (word_length, letter_number), count in counter.items():
        totals_by_length[word_length] += count
        max_letter_by_length[word_length] = max(
            max_letter_by_length[word_length], letter_number
        )

    rows = []

    # iterate through ALL positions, not just observed ones
    for word_length in sorted(totals_by_length.keys()):
        total = totals_by_length[word_length]

        # IMPORTANT: use full word length range
        for letter_number in range(word_length):
            count = counter.get((word_length, letter_number), 0)
            value = count / total if total > 0 else 0.0

            rows.append({
                "word_length": word_length,
                "letter_number": letter_number,
                value_name: value
            })

    return pd.DataFrame(rows)



def main():
    logs = load_logs(INPUT_JSON)

    forward_multiple_counts = defaultdict(int)
    forward_single_counts = defaultdict(int)
    intraword_regression_counts = defaultdict(int)

    for ep in logs:
        word_len = int(ep["word_len"])
        actions = fixation_actions_from_episode(ep)

        if not actions:
            continue

        # single-fixation vs multiple-fixation words
        if len(actions) == 1:
            count_positions(actions, forward_single_counts, word_len)
        else:
            count_positions(actions, forward_multiple_counts, word_len)

        # intraword regressions only:
        # landing position of any fixation that moves leftward within the word
        for prev_a, curr_a in zip(actions[:-1], actions[1:]):
            if curr_a < prev_a:
                intraword_regression_counts[(word_len, curr_a)] += 1

    df_forward_multiple = counts_to_proportion_df(
        forward_multiple_counts, "proportion_of_fixation"
    )
    df_forward_single = counts_to_proportion_df(
        forward_single_counts, "proportion_of_fixation"
    )
    df_intraword_reg = counts_to_proportion_df(
        intraword_regression_counts, "probability_of_regression"
    )

    df_forward_multiple.to_csv(OUT_FORWARD_MULTIPLE, index=False)
    df_forward_single.to_csv(OUT_FORWARD_SINGLE, index=False)
    df_intraword_reg.to_csv(OUT_REG_INTRAWORD, index=False)

    print("Saved:")
    print(" -", OUT_FORWARD_MULTIPLE)
    print(" -", OUT_FORWARD_SINGLE)
    print(" -", OUT_REG_INTRAWORD)


if __name__ == "__main__":
    main()
