from typing import Dict

import numpy as np

from coordination.common.types import ParameterValueType
from coordination.metadata.metadata import Metadata
from coordination.common.constants import DEFAULT_NUM_TIME_STEPS, DEFAULT_NUM_SUBJECTS


class ModelConfigBundle:
    """
    Container with different parameters of a model.
    """

    num_subjects: int = DEFAULT_NUM_SUBJECTS
    num_time_steps_in_coordination_scale: int = DEFAULT_NUM_TIME_STEPS
    perc_time_steps_to_fit: float = 1.0

    def update(self, params_dict: Dict[str, ParameterValueType]):
        """
        Update object attributes with values from a dictionary.

        @param params_dict: dictionary with attribute values. The keys must match the attribute
            names.
        """
        if params_dict is None:
            return

        for key, value in params_dict.items():
            if hasattr(self, key):
                if isinstance(value, list):
                    if isinstance(getattr(self, key), np.ndarray):
                        value = np.array(value)
                setattr(self, key, value)
