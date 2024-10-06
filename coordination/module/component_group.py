from __future__ import annotations

from typing import Dict, List, Optional

import pymc as pm

from coordination.module.latent_component.latent_component import \
    LatentComponent
from coordination.module.module import Module, ModuleSamples
from coordination.module.observation.observation import Observation
from coordination.module.transformation.transformation import Transformation


class ComponentGroup(Module):
    """
    This class represents a component group. It groups together a latent component and different
    observations associated with it.
    """

    def __init__(
            self,
            uuid: str,
            pymc_model: pm.Model,
            latent_component: Optional[LatentComponent],
            observations: List[Observation],
            transformations: Optional[List[Transformation]] = None,
            common_cause: Optional[LatentComponent] = None,
    ):
        """
        Creates a component group.

        @param uuid: string uniquely identifying the component group in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param latent_component: a latent system component.
        @param observations: a list of observations associated with the latent component.
        @param transformations: a list of transformations associated with the observations.
        @param common_cause: an optional common cause associated with the latent component.
        """
        super().__init__(
            uuid=uuid, pymc_model=pymc_model, parameters=None, observed_values=None
        )

        if transformations is not None:
            if len(transformations) != len(observations):
                raise ValueError(
                    f"The number of transformations ({len(transformations)}) does "
                    f"not match the number of observations ({len(observations)}."
                )

        if latent_component:
            self.latent_component = latent_component
        else:
            # Bypass for groups with observations that depend directly on coordination (e.g.,
            # spike observation)
            self.latent_component = LatentComponent(
                uuid="null_latent_component",
                pymc_model=pymc_model,
                parameters=None,
                num_subjects=0,
                dimension_size=0,
                self_dependent=False,
            )
        self.observations = observations
        self.transformations = transformations
        self.common_cause = common_cause

    def draw_samples(
            self, seed: Optional[int], num_series: int
    ) -> ComponentGroupSamples:
        """
        Draws latent component and observations samples using ancestral sampling and some blending
        strategy with coordination and different subjects.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: latent component and observation samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if self.common_cause:
            self.common_cause.coordination_samples = self.latent_component.coordination_samples
            common_cause_samples = self.common_cause.draw_samples(seed, num_series)
        else:
            common_cause_samples = None
        self.latent_component.common_cause_samples = common_cause_samples
        latent_component_samples = self.latent_component.draw_samples(seed, num_series)
        observation_samples = {}
        transformation_list = (
            [None] * len(self.observations)
            if self.transformations is None
            else self.transformations
        )
        for transformation, observation in zip(transformation_list, self.observations):
            observation.coordination_samples = (
                self.latent_component.coordination_samples
            )
            if transformation is None:
                transformed_latent_samples = latent_component_samples
            else:
                transformation.input_samples = latent_component_samples
                transformed_latent_samples = transformation.draw_samples(
                    seed, num_series
                )

            observation.latent_component_samples = transformed_latent_samples
            observation_samples[observation.uuid] = observation.draw_samples(
                seed, num_series
            )

        return ComponentGroupSamples(latent_component_samples,
                                     observation_samples,
                                     common_cause_samples)

    def create_random_variables(self):
        """
        Creates random variables for the latent component and associated observations.
        """
        super().create_random_variables()

        if self.common_cause:
            self.common_cause.create_random_variables()
            self.latent_component.common_cause_random_variable = \
                self.common_cause.latent_component_random_variable

        self.latent_component.create_random_variables()
        transformation_list = (
            [None] * len(self.observations)
            if self.transformations is None
            else self.transformations
        )
        for transformation, observation in zip(transformation_list, self.observations):
            if transformation is None:
                transformed_latent_rv = (
                    self.latent_component.latent_component_random_variable
                )
            else:
                transformation.input_random_variable = (
                    self.latent_component.latent_component_random_variable
                )
                transformation.create_random_variables()
                transformed_latent_rv = transformation.output_random_variable

            observation.coordination_random_variable = (
                self.latent_component.coordination_random_variable
            )
            observation.latent_component_random_variable = transformed_latent_rv
            observation.create_random_variables()


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class ComponentGroupSamples(ModuleSamples):
    def __init__(
            self,
            latent_component_samples: ModuleSamples,
            observation_samples: Dict[str, ModuleSamples],
            common_cause_samples: Optional[ModuleSamples] = None
    ):
        """
        Creates an object to store latent samples and samples from associates observations.

        @param latent_component_samples: samples generated by the latent component of the group.
        @param observation_samples: a dictionary of samples from each observation indexed by the
            observation module's id.
        @param common_cause_samples: optional samples generated by the common cause.
        """
        super().__init__(values=None)

        self.latent_component_samples = latent_component_samples
        self.observation_samples = observation_samples
        self.common_cause_samples = common_cause_samples
