from typing import Dict

import numpy as np

from coordination.common.types import ParameterValueType


class ModelConfigBundle:
    """
    Container with different parameters of a model.
    """

    def update(self, params_dict: Dict[str, ParameterValueType]):
        """
        Update object attributes with values from a dictionary.

        @param params_dict: dictionary with attribute values. The keys must match the attribute
            names.
        """

        for key, value in params_dict.items():
            if hasattr(self, key):
                if isinstance(value, list):
                    if isinstance(getattr(self, key), np.ndarray):
                        value = np.array(value)
                setattr(self, key, value)
