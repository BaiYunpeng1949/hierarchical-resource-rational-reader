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
  - Affects comprehension integration through confidence weighting
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
  - Reduces uncertainty through re-reading
- **Effect**: Improves integration of difficult content

## Implementation Details

### State Management
- **Word States**: Track for each word:
  - Embedding: Raw word representation
  - Comprehension: Processed understanding
  - Difficulty: Integration/uncertainty measure

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

## References

The implementation draws from cognitive theories of reading comprehension, including:
- Predictive processing in reading
- Integration difficulty measures
- Working memory constraints
- Eye movement control in reading 