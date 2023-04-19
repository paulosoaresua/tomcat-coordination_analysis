from __future__ import annotations
from typing import Any, Callable

import numpy as np
import pymc as pm
from scipy.special import softmax

from coordination.common.functions import sigmoid


class ActivationFunction:
    NAME_TO_NUMBER = {
        "linear": 0,
        "sigmoid": 1,
        "softmax": 2,
        "tanh": 3,
        "relu": 4,
    }

    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, x: Any) -> Any:
        return self.fn(x)

    @classmethod
    def from_pytensor_name(cls, name: str) -> Any:
        try:
            return cls.from_pytensor_number(cls.NAME_TO_NUMBER.get(name, -1))
        except:
            raise Exception(f"Invalid activation function {name}")

    @classmethod
    def from_pytensor_number(cls, number: int) -> ActivationFunction:
        if number == 0:
            return cls(lambda x: x)
        elif number == 1:
            return cls(pm.math.sigmoid)
        elif number == 2:
            return cls(pm.math.softmax)
        elif number == 3:
            return cls(pm.math.tanh)
        elif number == 4:
            return cls(lambda x: pm.math.maximum(x, 0))
        else:
            raise Exception(f"Invalid activation function index {number}")

    @classmethod
    def from_numpy_name(cls, name: str) -> Any:
        try:
            return cls.from_numpy_number(cls.NAME_TO_NUMBER.get(name, -1))
        except:
            raise Exception(f"Invalid activation function {name}")

    @classmethod
    def from_numpy_number(cls, index: int) -> ActivationFunction:
        if index == 0:
            return cls(lambda x: x)
        elif index == 1:
            return cls(sigmoid)
        elif index == 2:
            return cls(softmax)
        elif index == 3:
            return cls(np.tanh)
        elif index == 4:
            return cls(lambda x: np.maximum(x, 0))
        else:
            raise Exception(f"Invalid activation function index {index}")
