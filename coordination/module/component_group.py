from typing import Tuple, Union, Optional

import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.module import Module, ModuleParameters, ModuleSamples
from coordination.module.latent_component import LatentComponent
from coordination.module.observation2 import Observation


class ComponentGroup(Module):
    """
    This class represents a component group. It groups together a latent component and different
    observations associated with it.
    """

    def __init__(self,
                 latent_component: LatentComponent,
                 observations: List[Observation]):
        """
        Creates a component group.

        @param latent_component: a latent system component.
        @param observations: a list of observations associated with the latent component.
        """
        super().__init__()

        self.latent_component = latent_component
        self.observations = observations

    def draw_samples(self,
                     seed: Optional[int],
                     coordination: CoordinationSamples = None,
                     **kwargs) -> ModuleSamples:

        """
        Draws latent component and observations samples using ancestral sampling and some blending
        strategy with coordination and different subjects.

        @param seed: random seed for reproducibility.
        @param coordination: sampled coordination values.
        @param kwargs: extra arguments to be defined by subclasses.
        @raise ValueError: if coordination is None.
        @return: latent component and observation samples for each coordination series.
        """

        if coordination is None:
            raise ValueError(f"No coordination samples.")

        latent_component_samples = self.latent_component.draw_samples(seed=seed,
                                                                      coordination=coordination)
        observation_samples = {}
        for observation in self.observations:
            observation_samples[observation.uuid] = observation.draw_samples(
                seed=seed,
                coordination=coordination,
                latent_component=latent_component_samples
            )

        # TODO: Return samples
        return None


def update_pymc_model(self, pymc_model: pm.Model, **kwargs) -> Tuple[
    Union[TensorTypes, pm.Distribution], ...]:

    self.latent_component.update_pymc_model(pymc_model=pymc_model)
    for observation in observations:
        # Set transformation
        observation.update_pymc_model(pymc_model=pymc_model)
