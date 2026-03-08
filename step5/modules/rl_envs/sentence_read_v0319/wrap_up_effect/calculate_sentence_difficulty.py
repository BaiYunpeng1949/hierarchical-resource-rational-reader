import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "gpt2"   # causal LM for proper surprisal
DATA_PATH = "data/simulation/all_words_regression_and_skip_probabilities.csv"
OUTPUT_PATH = "data/sentence_difficulty.csv"


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
# Compute surprisal
# -----------------------------
def compute_surprisal(sentence, target_word_index):
    """
    Compute surprisal of the target word given its left context.
    surprisal = -log P(word | context)
    """

    words = sentence.split()
    context = " ".join(words[:target_word_index])
    target_word = words[target_word_index]

    if context.strip() == "":
        context = tokenizer.bos_token or ""

    # Tokenize context
    inputs = tokenizer(context, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]   # prediction for next token

    probs = torch.softmax(logits, dim=-1)

    target_tokens = tokenizer(target_word, add_special_tokens=False)["input_ids"]

    # approximate probability for first token
    token_prob = probs[0, target_tokens[0]]

    surprisal = -torch.log(token_prob + 1e-10).item()

    return surprisal


# -----------------------------
# Load dataset
# -----------------------------
print("Loading simulation data...")

df = pd.read_csv(DATA_PATH)

# expected columns:
# sentence_index | word_index | word | regression_prob | skip_prob

sentences = {}

for _, row in df.iterrows():

    idx = row["sentence_id"]
    word = row["word"]

    # skip missing words
    if pd.isna(word):
        continue

    # convert everything to string
    word = str(word)

    if idx not in sentences:
        sentences[idx] = []

    sentences[idx].append(word)


# reconstruct sentence strings
sentences = {
    idx: " ".join(words)
    for idx, words in sentences.items()
}


# -----------------------------
# Compute difficulty
# -----------------------------
print("Computing sentence difficulty...")

records = []

for idx, sentence in tqdm(sentences.items()):

    words = sentence.split()

    final_word_index = len(words) - 1

    difficulty = compute_surprisal(sentence, final_word_index)

    records.append({
        "index": idx,
        "sentence": sentence,
        "difficulty": difficulty
    })


result_df = pd.DataFrame(records)

result_df = result_df.sort_values("index")

# -----------------------------
# Add normalized difficulty columns
# -----------------------------

# Z-score normalization
mean_diff = result_df["difficulty"].mean()
std_diff = result_df["difficulty"].std()

result_df["difficulty_zscore"] = (result_df["difficulty"] - mean_diff) / std_diff

# Min-max normalization (0–1)
min_diff = result_df["difficulty"].min()
max_diff = result_df["difficulty"].max()

result_df["difficulty_norm"] = (result_df["difficulty"] - min_diff) / (max_diff - min_diff)

# -----------------------------
# Save results
# -----------------------------

os.makedirs("data", exist_ok=True)

result_df.to_csv(OUTPUT_PATH, index=False)

print("Saved sentence difficulties to:", OUTPUT_PATH)