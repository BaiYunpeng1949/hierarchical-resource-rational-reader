import math
import numpy as np

from gymnasium.spaces import Dict, Discrete
from gymnasium import Wrapper


def calc_dynamic_text_comprehension_score(scores, mode="softmin", tau=0.1):
    """
    Compute the dynamic text comprehension score

    Args:
        scores: list[float] -- the appraisalscores of the sentences
        mode: str
        tau: float

    Returns:
        float
    """
    if mode == "geometric mean":
        log_sum = sum(math.log(max(s, 1e-9)) for s in scores)
        return math.exp(log_sum / len(scores))
    elif mode == "harmonic mean":
        return len(scores) / sum(1/s for s in scores)
    elif mode == "softmin":
        w = np.exp(-np.array(scores) / tau)
        return float((w * scores).sum() / w.sum())
    else:
        raise ValueError(f"Invalid mode: {mode}")


class DictActionUnwrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(env.action_space['action_type'].n * env.action_space['regress_target'].n)

        # Save structure
        self._action_type_n = env.action_space['action_type'].n
        self._regress_target_n = env.action_space['regress_target'].n
    
    def reset(self, *args, **kwargs):
        """Forward reset arguments (e.g. inputs) to the underlying env."""
        return self.env.reset(*args, **kwargs)

    def step(self, action, **kwargs):
        # Unflatten
        action_type = action // self._regress_target_n
        regress_target = action % self._regress_target_n
        action_dict = {
            'action_type': action_type,
            'regress_target': regress_target
        }
        return self.env.step(action_dict, **kwargs)


if __name__ == "__main__":
    # scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    scores = [np.inf, np.inf, np.inf]
    print(calc_dynamic_text_comprehension_score(scores, mode="softmin", tau=0.4))