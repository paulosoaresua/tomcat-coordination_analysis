from typing import List, Union

import numpy as np
from scipy.stats import beta as scipy_beta


def beta(mean: Union[float, np.ndarray], var: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Beta distribution parameterized by a mean and a standard deviation
    """
    c = mean * (1 - mean) / var - 1
    a = mean * c
    b = (1 - mean) * c
    return scipy_beta(a, b)
