from __future__ import annotations

from typing import Union

import numpy as np
import pytensor.tensor as ptt

TensorTypes = Union[np.ndarray, ptt.TensorVariable, ptt.TensorConstant]
ParameterValueType = Union[str, int, float, np.ndarray]
