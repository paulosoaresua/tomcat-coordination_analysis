from typing import Optional

import numpy as np
import pymc as pm

from coordination.model.config_bundle.fnirs import FNIRSConfigBundle
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.non_serial_2d_gaussian_latent_component import \
    NonSerial2DGaussianLatentComponent
from coordination.module.observation.non_serial_gaussian_observation import \
    NonSerialGaussianObservation
from coordination.module.transformation.mlp import MLP


class FNIRSModel(ModelTemplate):
    """
    This class represents a model of fNIRS signals as from subjects performing a specific task. It
    uses a 2D latent component comprised of a position and a speed dimension.
    """

    def __init__(
        self,
        config_bundle: FNIRSConfigBundle,
        pymc_model: Optional[pm.Model] = None,
    ):
        """
        Creates an fNIRS model.

        @param config_bundle: container for the different parameters of the vocalic model.
        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        """

        if not pymc_model:
            pymc_model = pm.Model()

        coordination = SigmoidGaussianCoordination(
            pymc_model=pymc_model,
            mean_mean_uc0=config_bundle.mean_mean_uc0,
            sd_mean_uc0=config_bundle.sd_mean_uc0,
            sd_sd_uc=config_bundle.sd_sd_uc,
            num_time_steps=config_bundle.num_time_steps_in_coordination_scale,
        )

        # Save a direct reference to state_space and observation for easy access in the parameter
        # setting functions in this class.
        self.state_space = NonSerial2DGaussianLatentComponent(
            uuid="state_space",
            pymc_model=pymc_model,
            num_subjects=config_bundle.num_subjects,
            mean_mean_a0=config_bundle.mean_mean_a0,
            sd_mean_a0=config_bundle.sd_mean_a0,
            sd_sd_a=config_bundle.sd_sd_a,
            share_mean_a0_across_subjects=config_bundle.share_mean_a0_across_subjects,
            share_sd_a_across_subjects=config_bundle.share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=config_bundle.share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=config_bundle.share_sd_a_across_dimensions,
            sampling_relative_frequency=config_bundle.sampling_relative_frequency,
        )

        # Transform latent samples before passing to the observation module to account for
        # non-linearity and/or different dimensions between the latent component and
        # associated observation
        self.transformation = MLP(
            uuid="state_space_to_signals_mlp",
            pymc_model=pymc_model,
            output_dimension_size=config_bundle.num_channels,
            mean_w0=config_bundle.mean_w0,
            sd_w0=config_bundle.sd_w0,
            num_hidden_layers=config_bundle.num_hidden_layers,
            hidden_dimension_size=config_bundle.hidden_dimension_size,
            activation=config_bundle.activation,
            axis=1,  # channel axis
        )

        self.observation = NonSerialGaussianObservation(
            uuid="signal",
            pymc_model=pymc_model,
            num_subjects=config_bundle.num_subjects,
            dimension_size=config_bundle.num_channels,
            dimension_names=config_bundle.channel_names,
            sd_sd_o=config_bundle.sd_sd_o,
            share_sd_o_across_subjects=config_bundle.share_sd_o_across_subjects,
            share_sd_o_across_dimensions=config_bundle.share_sd_o_across_dimensions,
            normalization=config_bundle.observation_normalization,
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=pymc_model,
            latent_component=self.state_space,
            observations=[self.observation],
            transformations=[self.transformation],
        )

        super().__init__(
            name="fnirs_model",
            pymc_model=pymc_model,
            config_bundle=config_bundle,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=config_bundle.coordination_samples,
        )

    def prepare_for_sampling(self):
        """
        Sets parameter values for sampling using values in the model's config bundle.
        """
        self.coordination.parameters.mean_uc0.value = (
            np.ones(1) * self.config_bundle.mean_uc0
            if self.config_bundle.mean_uc0 is not None
            else None
        )
        self.coordination.parameters.sd_uc.value = (
            np.ones(1) * self.config_bundle.sd_uc
            if self.config_bundle.sd_uc is not None
            else None
        )
        self.state_space.parameters.mean_a0.value = self.config_bundle.mean_a0
        self.state_space.parameters.sd_a.value = self.config_bundle.sd_a
        self.observation.parameters.sd_o.value = self.config_bundle.sd_o

        if self.transformation:
            for i, w in enumerate(self.transformation.parameters.weights):
                w.value = self.config_bundle.weights[i]

    def prepare_for_inference(self):
        """
        Sets parameter values for inference using values in the model's config bundle.
        """

        # Fill parameter values from config bundle. If values are provided for a parameter, that
        # parameter won't be latent.
        self.prepare_for_sampling()

        self.coordination.num_time_steps = (
            self.config_bundle.num_time_steps_in_coordination_scale
        )
        self.state_space.time_steps_in_coordination_scale = (
            self.config_bundle.time_steps_in_coordination_scale
        )
        self.observation.observed_values = self.config_bundle.observed_values
        self.observation.time_steps_in_coordination_scale = (
            self.config_bundle.time_steps_in_coordination_scale
        )
