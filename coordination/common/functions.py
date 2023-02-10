from typing import Any, Optional, Union

from datetime import datetime
import random

import numpy as np


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        to_json = getattr(obj, "to_json", None)
        if callable(to_json):
            return to_json()

    raise TypeError(f"Type {type(obj)} is not serializable")


def set_seed(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def safe_divide(x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    return np.where(np.isclose(y, tol), 0, x / y)


def logit(x: Union[np.ndarray, float]) -> Union[np.ndarray, float, Any]:
    return np.log(x / (1 - x))


def sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float, Any]:
    return np.exp(x) / (1 + np.exp(x))
