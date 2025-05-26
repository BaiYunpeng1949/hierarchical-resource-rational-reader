##Documentation
Adaptive reading strategy / reading actions: revisit the sentences with lower appraisal levels. Prioritize the sentences with lower appraisals first.

NOTE: something to improve: 
From the Kintsch's paper, he mentioned something like "The mechanism of only a limited number of propositions could be processed in one cycle due to the STM limit, is akin to having a reading strategy where only the most relevant or recently processed information is actively maintained, while the rest may be dropped if not reinforced by further processing."
Action: so later I could introduce a memory decay over sentences (in a higher level, not micro-propositions), showing the similar trend.
Warning: to make the above-mentioned effect clear, maybe need to make the agent's initial appraisals range from an overall higher region. 

##Tunable parameters
1. apply stm limit or not [Kintsch's];
2. apply memory gist or not [Kintsch's];
3. stm size s [Kintsch's].

##Reproduction
Procedure: 

1. cd step5/
2. python main.py
3. cd step5/modules/rl_envs/text_comprehension_v0516/utils
4. python plot.py