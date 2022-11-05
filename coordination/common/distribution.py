from typing import Union

import numpy as np
from scipy.stats import beta as scipy_beta


def beta(mean: Union[float, np.ndarray], s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Beta distribution parameterized by a mean and a standard deviation
    """
    return scipy_beta(mean / s + mean, (1 - mean) * (1 - s) / s)
