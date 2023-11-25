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
