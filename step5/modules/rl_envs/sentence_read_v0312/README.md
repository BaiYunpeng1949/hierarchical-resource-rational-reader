# Sentence Reading Environment (v0312)

This environment implements a cognitively-inspired model of sentence reading behavior, using neural language models to track comprehension and simulate human-like reading patterns.

## Architecture Overview

The environment uses a three-component architecture:
1. **Language Model (BERT)**: Converts words into contextual embeddings
2. **GRU Network**: Tracks cumulative comprehension as we read
3. **Uncertainty Estimation**: Guides reading behavior decisions

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
  - Aligns with cognitive theories of surprisal and prediction error
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
   - First n letters of upcoming words
   - Word length and shape features
   - Integration with predictability

2. **Enhanced Prediction**:
   - Combine context with preview information
   - Word frequency effects
   - More sophisticated uncertainty estimation

3. **Cognitive Alignment**:
   - Better modeling of working memory constraints
   - More realistic regression patterns
   - Integration with eye-tracking data

## Limitations

- Simplified word skipping mechanism
- Basic uncertainty estimation
- Limited working memory modeling
- No explicit syntax processing
- Unidirectional GRU (could be enhanced with bidirectional processing)

## References

The implementation draws from cognitive theories of reading comprehension, including:
- Predictive processing in reading
- Integration difficulty measures
- Working memory constraints
- Eye movement control in reading 