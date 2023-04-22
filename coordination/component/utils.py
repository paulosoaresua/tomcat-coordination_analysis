from typing import Any, Callable

import numpy as np
import pymc as pm
import pytensor as pt
import pytensor.tensor as ptt

from coordination.common.activation_function import ActivationFunction


def add_bias(X: Any):
    if isinstance(X, np.ndarray):
        return np.concatenate([X, np.ones((1, X.shape[-1]))], axis=0)
    else:
        return ptt.concatenate([X, ptt.ones((1, X.shape[-1]))], axis=0)
