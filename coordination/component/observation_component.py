from typing import Any, List, Optional

import numpy as np
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior


class ObservationComponentParameters:

    def __init__(self, sd_sd_o: np.ndarray):
        self.sd_o = Parameter(HalfNormalParameterPrior(sd_sd_o))

    def clear_values(self):
        self.sd_o.value = None


class ObservationComponent:
    """
    This class models generic observations. Use specific observation classes for non-serial and serial components
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dim_value: int,
                 sd_sd_o: np.ndarray,
                 share_sd_o_across_subjects: bool,
                 share_sd_o_across_features: bool):

        # Check dimensionality of the hyper-prior parameters
        if share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = dim_value

        if share_sd_o_across_subjects:
            assert (dim_sd_o_features,) == sd_sd_o.shape
        else:
            assert (num_subjects, dim_sd_o_features) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_features = share_sd_o_across_features

        self.parameters = ObservationComponentParameters(sd_sd_o)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self.sd_o_name
        ]

    @property
    def sd_o_name(self) -> str:
        return f"sd_o_{self.uuid}"
