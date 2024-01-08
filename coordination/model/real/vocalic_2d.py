from typing import Optional

import pymc as pm

from coordination.model.config_bundle.vocalic import Vocalic2DConfigBundle
from coordination.model.model import Model
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.serial_2d_gaussian_latent_component import \
    Serial2DGaussianLatentComponent
from coordination.module.observation.serial_gaussian_observation import \
    SerialGaussianObservation
from coordination.module.transformation.dimension_reduction import \
    DimensionReduction
from coordination.module.transformation.mlp import MLP
from coordination.module.transformation.sequential import Sequential


class Vocalic2DModel(ModelTemplate):
    """
    This class represents a vocalic model where subjects are talking to each other and their
    speech vocalics are observed as they finish talking. It uses a 2D latent component comprised
    of a position and a speed dimension.
    """

    def __init__(
        self,
        config_bundle: Vocalic2DConfigBundle,
        pymc_model: Optional[pm.Model] = None,
    ):
        """
        Creates a vocalic model.

        @param config_bundle: container for the different parameters of the vocalic model.
        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        """
        super().__init__(pymc_model, config_bundle)

    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """
        coordination = SigmoidGaussianCoordination(
            pymc_model=self.pymc_model,
            num_time_steps=self.config_bundle.num_time_steps_in_coordination_scale,
            mean_mean_uc0=self.config_bundle.mean_mean_uc0,
            sd_mean_uc0=self.config_bundle.sd_mean_uc0,
            sd_sd_uc=self.config_bundle.sd_sd_uc,
            mean_uc0=self.config_bundle.mean_uc0,
            sd_uc=self.config_bundle.sd_uc,
        )

        state_space = Serial2DGaussianLatentComponent(
            uuid="state_space",
            pymc_model=self.pymc_model,
            num_subjects=self.config_bundle.num_subjects,
            mean_mean_a0=self.config_bundle.mean_mean_a0,
            sd_mean_a0=self.config_bundle.sd_mean_a0,
            sd_sd_a=self.config_bundle.sd_sd_a,
            share_mean_a0_across_subjects=self.config_bundle.share_mean_a0_across_subjects,
            share_sd_a_across_subjects=self.config_bundle.share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=self.config_bundle.share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=self.config_bundle.share_sd_a_across_dimensions,
            coordination_samples=self.config_bundle.coordination_samples,
            sampling_time_scale_density=self.config_bundle.sampling_time_scale_density,
            allow_sampled_subject_repetition=self.config_bundle.allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=self.config_bundle.fix_sampled_subject_sequence,
            time_steps_in_coordination_scale=self.config_bundle.time_steps_in_coordination_scale,
            prev_time_same_subject=self.config_bundle.prev_time_same_subject,
            prev_time_diff_subject=self.config_bundle.prev_time_diff_subject,
            subject_indices=self.config_bundle.subject_indices,
            mean_a0=self.config_bundle.mean_a0,
            sd_a=self.config_bundle.sd_a,
        )

        transformation = Sequential(
            child_transformations=[
                DimensionReduction(keep_dimensions=[0], axis=0),  # position,
                MLP(
                    uuid="state_space_to_speech_vocalics_mlp",
                    pymc_model=self.pymc_model,
                    output_dimension_size=self.config_bundle.num_vocalic_features,
                    mean_w0=self.config_bundle.mean_w0,
                    sd_w0=self.config_bundle.sd_w0,
                    num_hidden_layers=self.config_bundle.num_hidden_layers,
                    hidden_dimension_size=self.config_bundle.hidden_dimension_size,
                    activation=self.config_bundle.activation,
                    axis=0,  # Vocalic features axis
                    weights=self.config_bundle.weights,
                ),
            ]
        )

        observation = SerialGaussianObservation(
            uuid="speech_vocalics",
            pymc_model=self.pymc_model,
            num_subjects=self.config_bundle.num_subjects,
            dimension_size=self.config_bundle.num_vocalic_features,
            sd_sd_o=self.config_bundle.sd_sd_o,
            share_sd_o_across_subjects=self.config_bundle.share_sd_o_across_subjects,
            share_sd_o_across_dimensions=self.config_bundle.share_sd_o_across_dimensions,
            normalization=self.config_bundle.observation_normalization,
            dimension_names=self.config_bundle.vocalic_feature_names,
            observed_values=self.config_bundle.observed_values,
            time_steps_in_coordination_scale=self.config_bundle.time_steps_in_coordination_scale,
            subject_indices=self.config_bundle.subject_indices,
            sd_o=self.config_bundle.sd_o,
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=[transformation],
        )

        self._model = Model(
            name="vocalic_2d_model",
            pymc_model=self.pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=self.config_bundle.coordination_samples,
        )
