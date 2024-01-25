from typing import Union

import numpy as np
from scipy.signal import find_peaks


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


def mean_at_peaks(x: np.ndarray, seconds: int) -> np.ndarray:
    """
    Computes the mean at local maxima points.

    @param x: time series of values.
    @param seconds: number of seconds for sustainable peaks.
    @return: mean at peaks.
    """
    peaks, _ = find_peaks(x, width=seconds)
    if len(peaks) > 0:
        return x[peaks].mean()
    else:
        return -1


def peaks_count(x: np.ndarray, seconds: int) -> int:
    """
    Computes the mean at local maxima points.

    @param x: time series of values.
    @param seconds: number of seconds for sustainable peaks.
    @return: mean at peaks.
    """
    peaks, _ = find_peaks(x, width=seconds)
    return len(peaks)