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


# Technical Details -- POMDP formulation
- State (S): 
  - (Ext): fixation position in the word, sampled letters, word length, word to recognize (related to U), (time) 
  - (Int): activated words in the lexical memory (holds a belief)
- Action (A):
  - (Ext): eye movement within the word (letter index)
  - (Int): continue or terminate and activate
- Obervation (O):
  - (Ext): fixation position in the word, sampled letters, word length, (time)
  - (Int): belief - probability distribution over all activated words
- Transition Function (T):
  - (Ext): static (fixations always apply to the desired place)
  - (Int): Bayesian Inference (including the approximated likelihood function p(w|sampled letters so far)) that generates beliefs over parallelly activated words
- Reward Function (R): r(t) = U + c(t)


# Run the code:
python main.py
copy paste csv files from prior_effects and word_length_effect into results/section1/simulated_results
python plot.py