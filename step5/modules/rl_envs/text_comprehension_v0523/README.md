Description:

    This is a testing version for propositions listing of memory representations.

    If this version works, we will close the project; if not, move to Kintsch's more complicated tree graph; or the knowledge graph.

Objective: 

    1. validate whether the agent, with only stacks of propositions, could capture the high- vs. low-knowledge's difference / effects on the text comprehension performance (text-base level evaluation, like proportional evaluation).

    2. validate whether the agent, with only stacks of propositions, could capture the high- vs. low-coherence's difference / effects on the text comprehension performance.

Evaluations:

    1. Proportional recall
        Calculation of Proportional Recall:
            1. The text was propositionalized using van Dijk & Kintsch’s (1983) framework, breaking it into:
                a. Micropropositions (detailed sentence-level content)
                b. Macropropositions (global or paragraph-level main ideas)
            2. Participants' recalls were scored for the presence of any part of a proposition.
            3. Calculation (normalised): # propositions recalled / total propositions in the text version

    2. Text-based questions
        Definition: These questions can be answered using a single sentence from the text. They require literal recall of explicitly stated information.
    
    3. Bridging Inference Questions [out of scope]
        Require connecting information across multiple sentences in the text.
        Example:
            “Which disease causes scarring of the heart valves?”
            You need to infer from different text segments that link rheumatic fever with valve scarring.
    
    4. Elaborative inference questions [out of scope]
        Require integrating textual information with prior knowledge.
        Example:
            “Explain why arrhythmia would be a threat to life.”
            Not directly stated; requires applying knowledge of heart function.
    
    5. Problem-Solving Questions [out of scope]
        Require applying multiple pieces of information to novel scenarios.
        Example:
            “What treatment would help a 28-year-old experiencing heart failure from a muscle disease?”
            Involves diagnosing from text info and reasoning about treatments.

**A basic testing method**: use the LLM to first construct a memory text, then redistill the propositions out of this memory text. Compare.

Estimated results: 

    1. by prompting the llm agent with high- or low-prior knowledge on the text, not much difference for the text comprehension tests, since all propositions are observable.

    2. by qa based on proposition stacks from high- or low-coherent texts, not much difference for the text comprehension tests, because there is no need for integration between propositions, all internal processings (e.g., inference) are done by the llm.

Reference paper: Are Good Texts Always Better? Interactions of Text Coherence, Background Knowledge, and Levels of Understanding in Learning from Text

