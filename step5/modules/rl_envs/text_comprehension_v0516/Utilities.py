import math
import numpy as np


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


if __name__ == "__main__":
    # scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    scores = [1, 0.2, 1]
    # scores = [1, 1, 0.2, 1, 1]
    print(calc_dynamic_text_comprehension_score(scores, mode="softmin", tau=0.4))