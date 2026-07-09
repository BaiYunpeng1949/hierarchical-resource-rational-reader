import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "gpt2"
DATA_PATH = "data/simulation/all_words_regression_and_skip_probabilities.csv"
OUTPUT_PATH = "data/sentence_ambiguity.csv"


# -----------------------------
# Load language model
# -----------------------------
print("Loading language model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


# -----------------------------
# Compute next-word entropy
# -----------------------------
def compute_next_word_entropy(context_text: str) -> float:
    """
    Compute entropy of the next-token distribution given a left context.
    Entropy is used here as a proxy for local ambiguity / uncertainty.

    H(P) = - sum_i p_i log p_i
    """
    if context_text.strip() == "":
        context_text = tokenizer.bos_token or ""

    inputs = tokenizer(context_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # next-token prediction

    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()

    return entropy


def compute_sentence_ambiguity(sentence: str) -> float:
    """
    Compute sentence ambiguity as the mean next-word entropy
    across all incremental contexts in the sentence.

    For sentence w1 ... wn, compute:
    mean_t H(P(w_{t+1} | w_1...w_t))
    """
    words = sentence.split()

    # Need at least 2 words to have one prediction step
    if len(words) < 2:
        return 0.0

    entropies = []

    for t in range(1, len(words)):
        context = " ".join(words[:t])
        entropy_t = compute_next_word_entropy(context)
        entropies.append(entropy_t)

    return sum(entropies) / len(entropies)


# -----------------------------
# Load dataset and reconstruct sentences
# -----------------------------
print("Loading simulation data...")

df = pd.read_csv(DATA_PATH)

# Adjust these if your CSV columns differ
sentence_col = "sentence_id"
word_col = "word"

# If word order is available, sort by it
if "word_id" in df.columns:
    df = df.sort_values([sentence_col, "word_id"])

sentences = {}

for _, row in df.iterrows():
    idx = row[sentence_col]
    word = row[word_col]

    if pd.isna(word):
        continue

    word = str(word)

    if idx not in sentences:
        sentences[idx] = []

    sentences[idx].append(word)

sentences = {
    idx: " ".join(str(w) for w in words if pd.notna(w))
    for idx, words in sentences.items()
}


# -----------------------------
# Compute ambiguity
# -----------------------------
print("Computing sentence ambiguity...")

records = []

for idx, sentence in tqdm(sentences.items()):
    ambiguity = compute_sentence_ambiguity(sentence)

    records.append({
        "index": idx,
        "sentence": sentence,
        "ambiguity": ambiguity
    })

result_df = pd.DataFrame(records)
result_df = result_df.sort_values("index").reset_index(drop=True)

# -----------------------------
# Add normalized columns
# -----------------------------
mean_amb = result_df["ambiguity"].mean()
std_amb = result_df["ambiguity"].std()

if std_amb == 0:
    result_df["ambiguity_zscore"] = 0.0
else:
    result_df["ambiguity_zscore"] = (result_df["ambiguity"] - mean_amb) / std_amb

min_amb = result_df["ambiguity"].min()
max_amb = result_df["ambiguity"].max()

if max_amb == min_amb:
    result_df["ambiguity_norm"] = 0.0
else:
    result_df["ambiguity_norm"] = (
        (result_df["ambiguity"] - min_amb) / (max_amb - min_amb)
    )

# -----------------------------
# Save
# -----------------------------
os.makedirs("data", exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

print("Saved sentence ambiguity to:", OUTPUT_PATH)
print(result_df[["ambiguity", "ambiguity_zscore", "ambiguity_norm"]].describe())