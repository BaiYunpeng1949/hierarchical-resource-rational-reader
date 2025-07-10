import json, numpy as np

def np_to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # Let the base class default method raise the TypeError
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")
