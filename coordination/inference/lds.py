from typing import Tuple

import numpy as np
from scipy.stats import norm


def apply_conditional_property(y: float, a: float, b: float, var_y: float, m: float, var_z: float):
    """
    Returns the mean and variance of the normal distribution of z proportional to
    N(y | a * z + b, var_y) * N(z | m, var_z)

    The normal distribution is:
    N(z | m + K * (y - (a * m + b)), (1 - K * a) * var_z), where K = (a * var_z) / (a^2 * var_z + var_c)
    """
    K = (a * var_z) / ((a ** 2) * var_z + var_y)
    mean = m + K * (y - (a * m + b))
    var = (1 - K * a) * var_z

    return mean, var


def apply_marginal_property(a: float, b: float, var_y: float, m: float, var_z: float):
    """
    Returns the mean and variance of the normal distribution of y equals to
    Integral[ N(y | a * z + b, var_y) * N(z | m, var_z) dz ]

    The normal distribution is:
    N(y | a * m + b, a^2 * var_z + var_y)
    """
    mean = a * m + b
    var = (a ** 2) * var_z + var_y

    return mean, var


def pdf_projection(mean: float, var: float, a: float, b: float) -> Tuple[float, float]:
    """
    Projects the mean and variance of a normal distribution to a truncated normal distribution.

    From Wikipedia: https://en.wikipedia.org/wiki/Truncated_normal_distribution
    Pdf Projection can be found in the book Optimal State Estimation by Dan Simon
    """

    if np.isclose(var, 0):
        return max(min(mean, 1), 0), var

    std = np.sqrt(var)
    alpha = (a - mean) / std
    beta = (b - mean) / std
    f = norm.pdf
    F = norm.cdf

    Z = F(beta) - F(alpha)
    shifted_mean = mean - std * (f(beta) - f(alpha) / Z)

    p1 = (beta * f(beta) - alpha * f(alpha)) / Z
    p2 = ((f(beta) - f(alpha)) / Z) ** 2
    shifted_var = var * (1 - p1 - p2)

    return shifted_mean, shifted_var

