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

        # # TODO debug delete later
        # # Check whether the w.sum() is 0
        # if w.sum() == 0:
        #     print(f"Detect w.sum() == 0, w: {w}, scores: {scores}")
        
        # # TODO debug delete later
        # # Check whether w is the nan or inf
        # if np.isnan(w).any():
        #     print(f"Detect nan in w, w: {w}, scores: {scores}")
        # elif np.isinf(w).any():
        #     print(f"Detect inf in w, w: {w}, scores: {scores}")
        
        # # TODO debug delete later
        # # Check whether the scores are the nan or inf
        # if np.isnan(scores).any():
        #     print(f"Detect nan in scores, scores: {scores}")
        # elif np.isinf(scores).any():
        #     print(f"Detect inf in scores, scores: {scores}")

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

    def step(self, action):
        # Unflatten
        action_type = action // self._regress_target_n
        regress_target = action % self._regress_target_n
        action_dict = {
            'action_type': action_type,
            'regress_target': regress_target
        }
        return self.env.step(action_dict)


if __name__ == "__main__":
    # scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    scores = [np.inf, np.inf, np.inf]
    print(calc_dynamic_text_comprehension_score(scores, mode="softmin", tau=0.4))