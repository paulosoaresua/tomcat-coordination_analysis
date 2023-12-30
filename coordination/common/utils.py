import random
from json import JSONEncoder
from typing import Optional

import numpy as np


def set_random_seed(seed: Optional[int]):
    """
    Sets a random seed for reproducibility.

    @param seed: seed.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


class NumpyArrayEncoder(JSONEncoder):
    """
    Encodes a numpy array into a list for serialization into a json.
    """

    def default(self, obj):
        """
        Converts object into a list if it is a numpy array instance.

        @param obj: object.
        @return: a serializable json object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return JSONEncoder.default(self, obj)


def adjust_dimensions(x: [np.array, float], num_rows: int, num_cols: int = 0) -> np.array:
    """
    Transforms an entry to an array of proper dimensions by repeating the values across rows and
    columns if the entry is a float.

    @param x: entry
    @param num_rows: size of the axis 0
    @param num_cols: size of the axis 1

    @return: entry with adjusted dimensions.
    """
    if x is None:
        return None

    if num_rows < 0 or num_cols < 0:
        raise ValueError(f"The number of rows ({num_rows}) and number of columns ({num_cols}) must "
                         f"be non-negative integers.")

    if isinstance(x, int):
        x = float(x)

    if isinstance(x, float):
        if num_rows == 0:
            return x

        x = np.ones(num_rows) * x
        if num_cols > 0:
            return x[:, None].repeat(num_cols, axis=1)
    else:
        if x.ndim > 2:
            raise ValueError(f"The entry x has more than two dimensions ({x.shape}).")

        if x.ndim == 1:
            if x.shape[0] != num_rows:
                raise ValueError(
                    f"The dimensions of x ({x.shape}) are incompatible with the requested "
                    f"dimensions ({num_rows}, {num_cols})")

        if x.ndim == 2:
            if x.shape[0] != num_rows or x.shape[1] != num_cols:
                raise ValueError(
                    f"The dimensions of x ({x.shape}) are incompatible with the requested "
                    f"dimensions ({num_rows}, {num_cols})")

    # Make sure the final array has float type.
    return x.astype(float)
