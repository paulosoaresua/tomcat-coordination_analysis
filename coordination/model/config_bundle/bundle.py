from typing import Dict, Union

import numpy as np

from coordination.common.types import ParameterValueType
from coordination.metadata.metadata import Metadata


class ModelConfigBundle:
    """
    Container with different parameters of a model.
    """

    num_subjects: int = 3
    num_time_steps_in_coordination_scale: int = 100

    # Used for both sampling and inference
    num_time_steps_to_fit: int = None

    # Fixed value of coordination during inference
    observed_coordination_for_inference: Union[float, np.ndarray] = None

    # Initial sampled values of coordination. In a call to draw samples, the model will start by
    # using the samples given here and move forward sampling in time until the number of time
    # steps desired have been reached.
    initial_coordination_samples: np.ndarray = None

    # Whether to use a constant model of coordination
    constant_coordination: bool = False

    # Parameters for inference
    mean_mean_uc0: float = 0.0  # Variable coordination
    sd_mean_uc0: float = 1.0
    sd_sd_uc: float = 1.0
    alpha_c: float = 1.0  # Fix coordination
    beta_c: float = 1.0

    # Given parameter values. Required for sampling, not for inference.
    mean_uc0: float = 0.0  # Coordination = 0.5
    sd_uc: float = 1.0

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
