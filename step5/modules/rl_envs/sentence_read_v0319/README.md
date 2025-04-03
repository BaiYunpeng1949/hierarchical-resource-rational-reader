# Sentence Reading Environment (v0319)

This environment implements a cognitively-inspired model of sentence reading behavior, using neural language models to track comprehension and simulate human-like reading patterns.

## Architecture Overview

The environment uses a three-component architecture:
1. **Language Model (BERT)**: Converts words into contextual embeddings
2. **GRU Network**: Tracks cumulative comprehension as we read
3. **Uncertainty Estimation**: Guides reading behavior decisions

## Data Processing Pipeline

### 1. Integration Difficulty Computation
- **Method**: Uses BERT's masked language modeling with bidirectional context
- **Computation**:
  - Masks target word in full sentence context
  - Computes surprisal: -log P(word | context_left, context_right)
  - Converts surprisal to difficulty score using sigmoid function
- **Rationale**:
  - Bidirectional context captures both preceding and following information
  - Surprisal measures how unexpected a word is in its context
  - Sigmoid scaling provides normalized difficulty scores [0,1]
- **Output Format**:
  ```json
  {
    "word": "example",
    "surprisal": 2.5,
    "integration_difficulty": 0.7
  }
  ```

### 2. Word Prediction System
- **Preview Mechanism**:
  - Uses first 2 letters for clear preview
  - Implements noisy preview matching for subsequent letters
  - Similar-looking letters are considered interchangeable
- **Length Estimation**:
  - Blurry length estimation based on word length
  - Tolerance increases with word length
  - Short words (≤4): ±1 letter
  - Medium words (≤6): ±2 letters
  - Long words (≤8): ±3 letters
  - Very long words (>8): ±(length/3) letters
- **Similar Letter Groups**:
  - Round letters: a/e/o
  - Vertical lines: i/l/1
  - Ascending/descending: h/b, p/q
  - Common confusions: u/n, c/e, v/w, r/n
- **Noise in Preview**:
  - First 2 letters: Exact match required
  - Next 3 letters: Increasing noise threshold (30% per position)
  - Random acceptance based on position
- **Output Format**:
  ```json
  {
    "word": "example",
    "next_word_predicted": "word",
    "predictability": 0.8,
    "prediction_metadata": {
      "preview_letters": 2,
      "clear_preview": "ex",
      "target_length": 7,
      "length_tolerance": {
        "min": 5,
        "max": 9
      }
    },
    "prediction_candidates": [
      {"word": "word", "probability": 0.8},
      {"word": "words", "probability": 0.1},
      {"word": "world", "probability": 0.05},
      {"word": "work", "probability": 0.03},
      {"word": "would", "probability": 0.02}
    ]
  }
  ```

### 3. Processing Pipeline
1. **Initial Processing**:
   - Load raw sentence dataset
   - Compute integration difficulty scores
   - Save intermediate results

2. **Prediction Processing**:
   - Load dataset with integration scores
   - Compute word predictions with preview
   - Add prediction metadata and candidates
   - Save final processed dataset

3. **Quality Control**:
   - Filter out special tokens and duplicates
   - Handle subword tokenization
   - Normalize probabilities
   - Provide fallback for no predictions

## Key Mechanisms

### 1. Comprehension Tracking
- **Two-Layer GRU**:
  - Layer 1: Captures local word relationships and immediate context
  - Layer 2: Processes higher-level sentence meaning and global context
- **State Representation**:
  - Hidden states maintain cumulative understanding
  - Each layer processes different levels of meaning
  - Shape: [num_layers=2, batch_size=1, hidden_size=768]
- **Comprehension Measurement**:
  - Global comprehension: Weighted average of processed word states
  - Exponential weighting scheme:
    - Recent words receive higher weights (recency effect)
    - Earlier context maintains influence but with reduced weight
    - Prevents comprehension dilution with sentence length
  - Vector norm as comprehension strength indicator:
    - Larger norm = stronger/more confident understanding
    - Smaller norm = weaker/less certain understanding
  - Cognitive plausibility:
    - Recency effect: Aligns with human working memory, where recent information is more accessible
    - Cumulative processing: Maintains influence of earlier context while emphasizing new information
    - Dynamic updating: Comprehension grows with each word rather than being diluted
    - Working memory constraints: Natural decay of older information without complete loss

### Individual Differences in Reading
- **Random GRU Initialization**:
  - Each environment instance initializes GRU weights randomly
  - Results in slightly different comprehension values for the same word
  - Models individual differences in reading comprehension
- **Cognitive Plausibility**:
  - Different readers process the same text differently
  - Initial mental states vary between reading sessions
  - Base comprehension strength varies across individuals
  - Aligns with empirical observations of reading variability
- **Implementation Details**:
  - Non-deterministic processing through random GRU initialization
  - Same word can yield different comprehension values:
    - Across different reading sessions
    - Even with identical context and reading actions
  - Maintains unpredictability in reading behavior
- **Benefits**:
  - Natural modeling of reader variability
  - Avoids overly deterministic text processing
  - Captures session-to-session reading variations
  - Enables exploration of individual reading strategies

### 2. Integration Difficulty
- **Computation**: Euclidean distance between:
  - Processed comprehension (GRU's global state)
  - Raw context (averaged word embeddings)
- **Rationale**:
  - Measures how much new information changes existing understanding
  - Aligns with cognitive theories of surprisal and prediction error -- There’s a large body of psycholinguistics/computational linguistics research showing that language-model surprisal correlates quite well with human reading difficulty metrics (eye-tracking data, self-paced reading times, etc.). Psycholinguistic Basis: A large body of research shows that reading times, fixation durations, and neural signals (EEG N400 amplitude) often correlate well with LM-based surprisal.
  - Example: "The man bit the dog" - semantically related but difficult to integrate

### 3. Word Skipping Mechanism
- **Prediction**:
  - Uses context window to predict skipped word meaning
  - Weights context by distance (more recent words → higher weight)
  - Modulated by word predictability
- **Uncertainty Estimation**:
  - Based on cosine similarity between predicted and context states
  - Range [0,1]: 0=certain, 1=uncertain
  - Affects comprehension integration through confidence weighting:
    - High confidence → stronger contribution to global state
    - Low confidence → reduced impact on comprehension
- **Dual State Update**:
  1. Skipped Word:
     - Predicted meaning from context
     - Uncertainty-weighted comprehension
  2. Next Word:
     - Normal reading comprehension update
     - Integration with accumulated context

### 4. Regression Mechanism
- **Trigger**: High uncertainty or integration difficulty
- **Process**:
  - Returns to previous word
  - Recomputes comprehension with updated context
  - Strengthens representation by combining:
    - Original word embedding
    - Full context (including words that triggered regression)
- **Effect**: 
  - Improves integration of difficult content
  - Strengthens overall comprehension through bidirectional context
  - Increases confidence through multiple processing passes

## Implementation Details

### State Management
- **Word States**: Track for each word:
  - Embedding: Raw word representation
  - Comprehension: Processed understanding (GRU state)
  - Difficulty: Integration/uncertainty measure
- **Global Comprehension**:
  - Maintained as running average of processed states
  - Reflects overall sentence understanding
  - Updates dynamically with each reading action

### Context Processing
- **Window Size**: Configurable context window
- **Integration**: 
  - Compares processed vs. raw context
  - Handles local and global comprehension
- **Memory**: Optional decay mechanism for distant context

## Future Improvements

1. **Parafoveal Preview**:
   - Adjustable preview length (currently fixed at 2)
   - More sophisticated letter similarity groups
   - Integration with word frequency effects

2. **Enhanced Prediction**:
   - Dynamic noise thresholds based on word length
   - Better handling of compound words
   - Improved candidate filtering

3. **Cognitive Alignment**:
   - Individual differences in preview accuracy
   - Word frequency effects on prediction
   - More realistic length estimation

## Limitations

- Fixed preview length (2 letters)
- Simplified letter similarity groups
- Basic noise threshold model
- Limited handling of compound words
- No word frequency effects

## References

The implementation draws from cognitive theories of reading comprehension, including:
- Predictive processing in reading
- Integration difficulty measures
- Working memory constraints
- Eye movement control in reading
- Parafoveal preview effects
- Word length estimation in reading 

## Experiment Logs
PPO87 (f63f13e) has the best skipping probability (best alignment to human data), but no regressions leared. A hacky version if without regressions.
PPO88 (97d66b8) now learns regressions, but the skippings are worse. But a usable version.
PPO89 (24095d5) not better regressions, but the skippings are even worse compared to ppo88.

## Analysis
So now our explanation to the skipping totally works. But for regression, we need a better problem framing.

## Improvement method:
- [High priority] Change the regression's benefit (either give more rewards, or cognitively, gain more information).
- [Low priority] Change the regression mechanism. Not simply regress the previous word because its integration value is not high.
- [Medium priority] For generalizability: use the normalized rank rather than bins to represent words integration values.
