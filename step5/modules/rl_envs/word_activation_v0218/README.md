# Word Recognition Environment

This environment simulates human-like word recognition behavior using a Bayesian Reader model. It focuses on the cognitive aspects of word recognition **without** relying on computer vision.

## Key Features

- **Bayesian Word Recognition**: Implements a probabilistic model of word recognition based on:
  - Letter sampling with foveal vision simulation
  - Belief distribution updates using Bayesian inference
  - Word activation based on posterior probabilities

- **Human-like Reading Characteristics**:
  - Foveal vision simulation with configurable foveal size
  - Parallel word activation with top-k candidates
  - Dynamic belief updates as more letters are sampled
  - Stochastic vs deterministic word activation options

## Important Notes

- **No Visual Perception**: This environment does NOT include computer vision or visual perception. It operates on:
  - Pre-defined word representations
  - Normalized character codes
  - Probabilistic letter sampling

- **Input Format**: Words are provided as:
  - Character sequences
  - Normalized ASCII codes
  - Pre-processed word representations

## Limitations

- No actual visual input processing
- Simplified letter sampling mechanism
- Fixed lexicon-based word recognition
- Limited to ASCII character set

## Future Improvements

1. Integration with language models for better word prediction
2. Support for non-ASCII characters
3. Dynamic lexicon updates
4. More sophisticated letter sampling strategies