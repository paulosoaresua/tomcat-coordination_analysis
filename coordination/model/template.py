from abc import ABC, abstractmethod
from typing import List, Optional

import pymc as pm

from coordination.model.config.bundle import ModelConfigBundle
from coordination.model.model import Model
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.coordination import Coordination
from coordination.module.module import ModuleSamples


class ModelTemplate(Model, ABC):
    """
    This class represents a template for concrete models. It extends the model class and
    incorporates helper functions to be called to set specific parameters needed for sampling and
    inference.
    """

    def __init__(
        self,
        name: str,
        pymc_model: pm.Model,
        config_bundle: ModelConfigBundle,
        coordination: Coordination,
        component_groups: List[ComponentGroup],
        coordination_samples: Optional[ModuleSamples] = None,
    ):
        """
        Creates a model template instance.

        @param name: a name for the model.
        @param pymc_model: a PyMC model instance where model's modules are to be created at.
        @param config_bundle: a config bundle with values for the different parameters of the
            model.
        @param coordination: coordination module.
        @param component_groups: list of component groups in the model.
        @param coordination_samples: fixed coordination samples to be used during a call to
            draw_samples. If provided, these samples will be used and samples from the coordination
            component won't be drawn.
        """

        super().__init__(
            name=name,
            pymc_model=pymc_model,
            coordination=coordination,
            component_groups=component_groups,
            coordination_samples=coordination_samples,
        )
        self.config_bundle = config_bundle

    @abstractmethod
    def prepare_for_sampling(self):
        """
        Sets parameter values for sampling using values in the model's config bundle.
        """

    @abstractmethod
    def prepare_for_inference(self):
        """
        Sets parameter values for inference using values in the model's config bundle.
        """
