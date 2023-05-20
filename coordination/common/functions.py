from typing import Any, Union

import numpy as np


def logit(x: Union[np.ndarray, float]) -> Union[np.ndarray, float, Any]:
    return np.log(x / (1 - x))


def sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float, Any]:
    return np.exp(x) / (1 + np.exp(x))
