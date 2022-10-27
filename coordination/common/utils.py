from typing import Optional

from datetime import datetime
import random

import numpy as np


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime):
        return obj.isoformat()

    raise TypeError(f"Type {type(obj)} is not serializable")


def set_seed(seed: Optional[int]):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
