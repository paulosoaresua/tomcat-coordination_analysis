from __future__ import annotations
from typing import Any, Callable

import pymc as pm


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
    def from_name(cls, name: str) -> Any:
        try:
            return cls.from_number(cls.NAME_TO_NUMBER.get(name, -1))
        except:
            raise Exception(f"Invalid activation function {name}")

    @classmethod
    def from_number(cls, index: int) -> ActivationFunction:
        if index == 0:
            return cls(lambda x: x)
        elif index == 1:
            return cls(pm.math.sigmoid)
        elif index == 2:
            return cls(pm.math.softmax)
        elif index == 3:
            return cls(pm.math.tanh)
        elif index == 4:
            return cls(lambda x: pm.math.maximum(x, 0))
        else:
            raise Exception(f"Invalid activation function index {index}")
