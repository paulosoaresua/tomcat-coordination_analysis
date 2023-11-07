from __future__ import annotations

from typing import Union

import numpy as np
import pytensor.tensor as ptt

TensorTypes = Union[np.array, ptt.TensorVariable, ptt.TensorConstant]
