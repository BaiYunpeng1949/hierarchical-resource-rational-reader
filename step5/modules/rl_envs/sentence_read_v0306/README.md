# Sentence Reading Environment (v0306)

This environment implements a simplified model of sentence reading behavior, focusing on comprehension dynamics without actual word-level processing. It uses pseudo-sentences represented as sequences of predictability states.

## Key Concepts

- **Pseudo-Sentences**: Instead of actual words, sentences are represented as:
  - Sequences of predictability values (0-1)
  - Each value represents a word's expected predictability
  - No actual word content or meaning

- **Comprehension Model**:
  - Tracks comprehension through appraisal values
  - Comprehension depends on:
    - Word predictability
    - Context from previous words
    - Integration with previous content

## Reading Behaviors

The environment simulates three main reading actions:

1. **Reading**:
   - Sequential word processing
   - Comprehension decay over time
   - Context-dependent understanding

2. **Skipping**:
   - Based on word predictability
   - Risk of comprehension loss
   - Context quality influence

3. **Regression**:
   - Triggered by low comprehension
   - Improves previous word understanding
   - Affects integration with context

## Important Notes

- **Simplified Approach**: This is a highly simplified model that:
  - Does not process actual words or text
  - Uses numerical states instead of linguistic content
  - Focuses on reading dynamics rather than language understanding

- **State Representation**:
  - Words are represented by predictability values
  - Comprehension is tracked through appraisal scores
  - Context is modeled through sliding windows

## Limitations

- No actual language processing
- Simplified comprehension model
- Binary success/failure outcomes
- Limited context modeling

## Future Improvements

1. Integration with language models for real text processing
2. More sophisticated comprehension modeling
3. Better context integration
4. Support for complex sentence structures 