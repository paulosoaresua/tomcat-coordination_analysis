from typing import Union

import numpy as np


def logit(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Computes the logit of a number or array.

    @param x: number of array to compute the logit for.
    @return: logit of x.
    """
    return np.log(x / (1 - x))


def sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Computes the sigmoid of a number or array.

    @param x: number of array to compute the sigmoid for.
    @return: sigmoid of x.
    """
    return np.exp(x) / (1 + np.exp(x))
