from typing import Optional

import numpy as np
import pymc as pm

from coordination.model.config_bundle.vocalic_semantic_link import \
    Vocalic2DSemanticLinkConfigBundle
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.null_latent_component import \
    NullLatentComponent
from coordination.module.latent_component.serial_first_derivative_latent_component import \
    SerialFirstDerivativeLatentComponent
from coordination.module.observation.serial_gaussian_observation import \
    SerialGaussianObservation
from coordination.module.observation.spike_observation import SpikeObservation
from coordination.module.transformation.mlp import MLP


class VocalicSemanticLinkModel(ModelTemplate):
    """
    This class represents a model where subjects are talking to each other and their speech
    vocalics are observed as they finish talking as well as semantic links between subsequent
    utterances from different subjects. It uses a 2D latent component comprised of a position and
    a speed dimension for the vocalic component.
    """

    def __init__(
        self,
        config_bundle: Vocalic2DSemanticLinkConfigBundle,
        pymc_model: Optional[pm.Model] = None,
    ):
        """
        Creates a vocalic + semantic link model.

        @param config_bundle: container for the different parameters of the vocalic + semantic
            link model.
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
        self.state_space = SerialFirstDerivativeLatentComponent(
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
            sampling_time_scale_density=config_bundle.sampling_time_scale_density,
            allow_sampled_subject_repetition=config_bundle.allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=config_bundle.fix_sampled_subject_sequence,
        )

        self.transformation = MLP(
            uuid="state_space_to_speech_vocalics_mlp",
            pymc_model=pymc_model,
            output_dimension_size=config_bundle.num_vocalic_features,
            mean_w0=config_bundle.mean_w0,
            sd_w0=config_bundle.sd_w0,
            num_hidden_layers=config_bundle.num_hidden_layers,
            hidden_dimension_size=config_bundle.hidden_dimension_size,
            activation=config_bundle.activation,
            axis=0,  # Vocalic features axis
        )

        self.observed_vocalics = SerialGaussianObservation(
            uuid="speech_vocalics",
            pymc_model=pymc_model,
            num_subjects=config_bundle.num_subjects,
            dimension_size=config_bundle.num_vocalic_features,
            dimension_names=config_bundle.vocalic_feature_names,
            sd_sd_o=config_bundle.sd_sd_o,
            share_sd_o_across_subjects=config_bundle.share_sd_o_across_subjects,
            share_sd_o_across_dimensions=config_bundle.share_sd_o_across_dimensions,
            normalization=config_bundle.normalize_observed_values,
        )

        vocalic_group = ComponentGroup(
            uuid="vocalic_group",
            pymc_model=pymc_model,
            latent_component=self.state_space,
            observations=[self.observed_vocalics],
            transformations=[self.transformation],
        )

        self.observed_semantic_links = SpikeObservation(
            uuid="semantic_link",
            pymc_model=pymc_model,
            num_subjects=config_bundle.num_subjects,
            a_p=config_bundle.a_p,
            b_p=config_bundle.b_p,
            dimension_name="linked",
            sampling_time_scale_density=config_bundle.sampling_time_scale_density,
        )

        semantic_link_group = ComponentGroup(
            uuid="semantic_link_group",
            pymc_model=pymc_model,
            latent_component=NullLatentComponent(),
            observations=[self.observed_semantic_links],
        )

        super().__init__(
            name="vocalic_semantic_link_model",
            pymc_model=pymc_model,
            config_bundle=config_bundle,
            coordination=coordination,
            component_groups=[vocalic_group, semantic_link_group],
            coordination_samples=config_bundle.coordination_samples,
        )

    def prepare_for_sampling(self):
        """
        Sets parameter values for sampling using values in the model's config bundle.
        """
        self.coordination.parameters.mean_uc0.value = (
            np.ones(1) * self.config_bundle.mean_uc0
            if self.config_bundle.mean_uc0
            else None
        )
        self.coordination.parameters.sd_uc.value = (
            np.ones(1) * self.config_bundle.sd_uc if self.config_bundle.sd_uc else None
        )
        self.state_space.parameters.mean_a0.value = self.config_bundle.mean_a0
        self.state_space.parameters.sd_a.value = self.config_bundle.sd_a
        self.observed_vocalics.parameters.sd_o.value = self.config_bundle.sd_o
        self.observed_semantic_links.parameters.p.value = self.config_bundle.p

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
            self.config_bundle.vocalics_time_steps_in_coordination_scale
        )
        self.state_space.subject_indices = self.config_bundle.subject_indices
        self.state_space.prev_time_same_subject = (
            self.config_bundle.prev_time_same_subject
        )
        self.state_space.prev_time_diff_subject = (
            self.config_bundle.prev_time_diff_subject
        )
        self.observed_vocalics.observed_values = (
            self.config_bundle.observed_vocalic_values
        )
        self.observed_vocalics.time_steps_in_coordination_scale = (
            self.config_bundle.vocalics_time_steps_in_coordination_scale
        )
        self.observed_vocalics.subject_indices = self.config_bundle.subject_indices
        self.observed_semantic_links.time_steps_in_coordination_scale = (
            self.config_bundle.semantic_link_time_steps_in_coordination_scale
        )
        self.observed_semantic_links.observed_values = np.ones_like(
            self.config_bundle.semantic_link_time_steps_in_coordination_scale
        )
