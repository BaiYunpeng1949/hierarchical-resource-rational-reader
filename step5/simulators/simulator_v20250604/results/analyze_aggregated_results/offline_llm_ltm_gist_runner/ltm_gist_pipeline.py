
import json
import re
from typing import Any, Dict, List, Optional

# import step5.utils.constants as const
from . import constants as const

from .LLMMemories import LLMLongTermMemory, LLMShortTermMemory, LLMWorkingMemory

def _join_tokens(tokens: List[str]) -> str:
    toks = [t for t in tokens or [] if isinstance(t, str) and t.strip()]
    s = " ".join(toks).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

class LTMPipelineByMemories:
    """
    Thin adapter that mirrors how `ReaderAgent` orchestrates schema activation,
    STM micro-facet extraction, and LTM macrostructure updates — but runs on
    offline JSON logs (no RL env).
    """

    def __init__(self, config_path: str, stm_capacity: int = 4):
        # Instantiate the same classes the simulator uses
        self.ltm = LLMLongTermMemory(config=config_path)
        self.stm = LLMShortTermMemory(config=config_path, stm_capacity=stm_capacity)
        self.wm  = LLMWorkingMemory(config=config_path)

        # Reset memory states like the simulator does on each trial
        self.ltm.reset()
        self.stm.reset()
        self.wm.reset()

    def process_sentence_cycle(self, sentence: str, sentence_id: int, reading_strategy: str = None):
        """Run ONE reading cycle: activate schemas -> STM facet -> LTM update."""
        if reading_strategy is None:
            reading_strategy = const.READ_STRATEGIES["normal"]

        # 1) Activate/refresh schemas based on raw sentence
        self.ltm.activate_schemas(raw_sentence=sentence)

        # 2) Extract microstructure and update STM (includes appraisal & visit count)
        self.stm.extract_microstructure_and_update_stm(
            raw_sentence=sentence,
            spatial_info=str(sentence_id),  # keep OrderedDict keys stable
            activated_schemas=self.ltm.activated_schemas,
            main_schemas=self.ltm.main_schemas,
            reading_strategy=reading_strategy
        )

        # 3) Update macrostructure in LTM from STM (schema‑aware relevance gating)
        self.ltm.generate_macrostructure(stm=self.stm.STM)

    def finalize(self) -> Dict[str, Any]:
        """Finalize LTM gists (updates root theme) and return a compact result dict."""
        self.ltm.finalize_gists()
        return {
            "main_schemas": self.ltm.main_schemas,
            "all_activated_schemas": self.ltm.all_activated_schemas,
            "outline": self.ltm.gists,   # string with bullet hierarchy
        }

def run_trial_from_logs(config_path: str, trial: Dict[str, Any], max_steps: Optional[int] = None) -> Dict[str, Any]:
    pipe = LTMPipelineByMemories(config_path=config_path)

    # TODO need to check whether we are running from a new agent every time, need to be a new agent

    logs = trial.get("text_reading_logs", [])
    for i, rec in enumerate(logs[: (max_steps or len(logs))]):
        # Reconstruct the sentence hypothesis from sampled tokens
        sentence = _join_tokens(rec.get("sampled_words_in_sentence", []))
        pipe.process_sentence_cycle(
            sentence=sentence,
            # sentence_id=rec.get("current_sentence_index", i),
            sentence_id=rec.get("actual_reading_sentence_index", i),
            reading_strategy=const.READ_STRATEGIES.get("normal", "normal")
        )

    out = pipe.finalize()
    out["episode_index"] = trial.get("episode_index")
    out["stimulus_index"] = trial.get("stimulus_index")
    out["time_condition"] = trial.get("time_condition")
    out["total_time"] = trial.get("total_time")
    return out
