from typing import Optional

import random

import numpy as np


def set_random_seed(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
