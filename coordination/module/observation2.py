from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.parametrization2 import (Parameter,
                                                  HalfNormalParameterPrior)
from coordination.module.module import Module, ModuleParameters, ModuleSamples
from coordination.module.coordination2 import CoordinationSamples
from coordination.module.latent_component import LatentComponentSamples


class Observation(ABC, Module):
    """
    This class represents an observation (O) from a latent system component (A). Observations are
    evidence to the model. This implementation samples observation from a Gaussian distribution
     centered on some transformation, g(.),  of the latent components, i.e., O ~ N(g(A), var_o).
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 dimension_names: Optional[List[str]] = None):
        """
        Creates an observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        """
        super().__init__()

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.dimension_names = dimension_names

        self.parameters = ObservationParameters(module_uuid=uuid,
                                                sd_sd_o=sd_sd_o)
