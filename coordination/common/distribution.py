import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm as scipy_truncnorm

from coordination.common.utils import safe_divide


def truncnorm(mean: np.ndarray, std: np.ndarray, a: float = 0, b: float = 1) -> scipy_truncnorm:
    # a = (a - mean) / std
    # b = (b - mean) / std
    # nominator = (norm().pdf(a) - norm().pdf(b)) * std
    # denominator = (norm().cdf(b) - norm().cdf(a))
    # offset = safe_divide(nominator, denominator)
    #
    # return scipy_truncnorm(loc=mean, scale=std, a=a, b=b)
    return norm(mean, std)
