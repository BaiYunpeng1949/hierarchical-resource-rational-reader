##Documentation
Adaptive reading strategy / reading actions: revisit the sentences with lower appraisal levels. Prioritize the sentences with lower appraisals first.

### Sentence Appraisal Generation
The sentence appraisals for real human stimuli are generated using a language model-based approach that captures cognitive integration difficulty:

1. **Surprisal-based Scoring**:
   - Uses GPT-2 model to compute surprisal (log loss) of each sentence
   - Takes into account prior context for each sentence
   - Lower surprisal indicates easier integration with context
   - Higher surprisal indicates more difficult integration

2. **Normalization Process**:
   - Scores are normalized per stimulus using min-max normalization
   - Preserves relative difficulty ranking within each text
   - Standardizes scores to [0,1] range for consistent interpretation
   - 0 represents hardest sentence in the text
   - 1 represents easiest sentence in the text

3. **Theoretical Justification**:
   - Based on Kintsch's Construction-Integration model (1988)
   - Reflects human comprehension as contextual and relative
   - Captures intra-text dynamics and coherence
   - Avoids scale drift across different texts
   - Maintains cognitive plausibility of relative difficulty assessment

Argument supports: 
1. Human Comprehension is Contextual and Relative
2. Min-Max Normalization Reflects Intra-Text Dynamics
3. Avoids Scale Drift Across Texts

NOTE: something to improve: 
From the Kintsch's paper, he mentioned something like "The mechanism of only a limited number of propositions could be processed in one cycle due to the STM limit, is akin to having a reading strategy where only the most relevant or recently processed information is actively maintained, while the rest may be dropped if not reinforced by further processing."
Action: so later I could introduce a memory decay over sentences (in a higher level, not micro-propositions), showing the similar trend.
Warning: to make the above-mentioned effect clear, maybe need to make the agent's initial appraisals range from an overall higher region. 

##Tunable parameters
1. apply stm limit or not [Kintsch's];
2. apply memory gist or not [Kintsch's];
3. stm size s [Kintsch's].
4. prior knowledge classification [Are Good Texts Alawys Better?] (currently by LLM's inputted prompts).
5. (low prority) propositions' integration and inference (currently we do not differentiate this, all done by LLM itself).

##Reproduction
Procedure: 

1. cd step5/
2. python main.py
3. cd step5/modules/rl_envs/text_comprehension_v0516/utils
4. python plot.py