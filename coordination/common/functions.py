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


def mean_at_peaks(series: np.ndarray, seconds: int) -> np.ndarray:
    """
    Computes the mean at local maxima points.

    @param series: series of values.
    @param seconds: number of seconds for sustainable peaks.
    @return: mean at peaks.
    """
    values = []

    for row in series:
        peaks, _ = find_peaks(row, width=seconds)
        if len(peaks) > 0:
            values.append(row[peaks].mean())
        else:
            values.append(0)

    return np.array(values)
