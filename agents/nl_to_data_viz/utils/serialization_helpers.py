import numpy as np
from decimal import Decimal
from typing import Any

def sanitize_for_serialization(obj: Any) -> Any:
    """Convert numpy/Decimal types to native Python types"""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {sanitize_for_serialization(k): sanitize_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_serialization(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_serialization(item) for item in obj)
    return obj