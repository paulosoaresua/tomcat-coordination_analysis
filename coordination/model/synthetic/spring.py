from typing import Optional

import pymc as pm

from coordination.common.utils import adjust_dimensions
from coordination.model.config_bundle.spring import SpringConfigBundle
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.non_serial_mass_spring_damper_latent_component import \
    NonSerialMassSpringDamperLatentComponent
from coordination.module.observation.non_serial_gaussian_observation import \
    NonSerialGaussianObservation
from coordination.module.transformation.mlp import MLP
from coordination.model.model import Model


class SpringModel(ModelTemplate):
    """
    This class represents a spring model where springs are influenced by each other as controlled
    by coordination.

    The variables in the latent component and observations are 2 dimensional. The first dimension
    contains the position of the mass attached to the spring and the second its speed. The speed
    evolves with the laws that govern harmonic oscillators but it is not blended by coordination
    as the mass position is.
    """

    def __init__(
            self,
            config_bundle: SpringConfigBundle,
            pymc_model: Optional[pm.Model] = None,
    ):
        """
        Creates a conversation model.

        @param config_bundle: container for the different parameters of the conversation model.
        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        """
        super().__init__(config_bundle=config_bundle, pymc_model=pymc_model)

    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """

        coordination = SigmoidGaussianCoordination(
            pymc_model=self.pymc_model,
            mean_mean_uc0=self.config_bundle.mean_mean_uc0,
            sd_mean_uc0=self.config_bundle.sd_mean_uc0,
            sd_sd_uc=self.config_bundle.sd_sd_uc,
            num_time_steps=self.config_bundle.num_time_steps_in_coordination_scale,
        )

        # Save a direct reference to state_space and observation for easy access in the parameter
        # setting functions in this class.
        state_space = NonSerialMassSpringDamperLatentComponent(
            uuid="state_space",
            pymc_model=self.pymc_model,
            num_subjects=self.config_bundle.num_springs,
            spring_constant=adjust_dimensions(
                self.config_bundle.spring_constant, num_rows=self.config_bundle.num_springs
            ),
            mass=adjust_dimensions(
                self.config_bundle.mass, num_rows=self.config_bundle.num_springs
            ),
            dampening_coefficient=adjust_dimensions(
                self.config_bundle.dampening_coefficient, num_rows=self.config_bundle.num_springs
            ),
            dt=self.config_bundle.time_step_size_in_seconds,
            mean_mean_a0=self.config_bundle.mean_mean_a0,
            sd_mean_a0=self.config_bundle.sd_mean_a0,
            sd_sd_a=self.config_bundle.sd_sd_a,
            share_mean_a0_across_subjects=self.config_bundle.share_mean_a0_across_springs,
            share_sd_a_across_subjects=self.config_bundle.share_sd_a_across_springs,
            share_mean_a0_across_dimensions=self.config_bundle.share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=self.config_bundle.share_sd_a_across_dimensions,
            sampling_relative_frequency=self.config_bundle.sampling_relative_frequency,
            time_steps_in_coordination_scale=(
                self.config_bundle.time_steps_in_coordination_scale
            ),
            mean_a0=self.config_bundle.mean_a0,
            sd_a=self.config_bundle.sd_a
        )

        transformation = None
        if self.config_bundle.observation_dim_size != 2:
            transformation = MLP(
                uuid="state_space_to_observation_mlp",
                pymc_model=self.pymc_model,
                output_dimension_size=self.config_bundle.observation_dim_size,
                mean_w0=self.config_bundle.mean_w0,
                sd_w0=self.config_bundle.sd_w0,
                num_hidden_layers=self.config_bundle.num_hidden_layers,
                hidden_dimension_size=self.config_bundle.hidden_dimension_size,
                activation=self.config_bundle.activation,
                axis=1,
                weights=self.config_bundle.weights,
            )

        observation = NonSerialGaussianObservation(
            uuid="observation",
            pymc_model=self.pymc_model,
            num_subjects=self.config_bundle.num_springs,
            dimension_size=self.config_bundle.observation_dim_size,
            sd_sd_o=self.config_bundle.sd_sd_o,
            share_sd_o_across_subjects=self.config_bundle.share_sd_o_across_springs,
            share_sd_o_across_dimensions=self.config_bundle.share_sd_o_across_dimensions,
            normalization=self.config_bundle.observation_normalization,
            observed_values=self.config_bundle.observed_values,
            time_steps_in_coordination_scale=(
                self.config_bundle.time_steps_in_coordination_scale
            ),
            sd_o=self.config_bundle.sd_o,
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=[transformation]
            if transformation is not None
            else None,
        )

        self._model = Model(
            name="spring_model",
            pymc_model=self.pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=self.config_bundle.coordination_samples
        )
