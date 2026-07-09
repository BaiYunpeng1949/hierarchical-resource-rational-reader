import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_CSV = "mcconkie.csv"
OUTPUT_CSV = "mcconkie_processed.csv"

# Column names (adjust if needed)
WORD_LEN_COL = "word_length"
LETTER_COL = "letter_number"
VALUE_COL = "proportion_of_fixation"   # or "percentage" depending on your file


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

# =========================
# STEP 1: DROP LETTER 0
# =========================
df = df[df[LETTER_COL] != 0].copy()

# =========================
# STEP 2: REINDEX LETTERS
# (original 1 → 0, 2 → 1, ...)
# =========================
df[LETTER_COL] = df[LETTER_COL] - 1

# =========================
# STEP 3: NORMALIZE WITHIN WORD LENGTH
# =========================
df[VALUE_COL] = df.groupby(WORD_LEN_COL)[VALUE_COL].transform(
    lambda x: x / x.sum()
)

# =========================
# OPTIONAL: SCALE TO PERCENTAGE (0–100)
# =========================
# Uncomment if you want percentages instead of probabilities
# df[VALUE_COL] = df[VALUE_COL] * 100

# =========================
# SORT (for clean plotting later)
# =========================
df = df.sort_values(by=[WORD_LEN_COL, LETTER_COL])

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved processed dataset to: {OUTPUT_CSV}")