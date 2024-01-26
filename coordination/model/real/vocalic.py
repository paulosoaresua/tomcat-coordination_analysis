from typing import Optional, List

import numpy as np
import pymc as pm

from coordination.model.config_bundle.vocalic import VocalicConfigBundle
from coordination.model.model import Model
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.serial_gaussian_latent_component import \
    SerialGaussianLatentComponent, SerialGaussianLatentComponentSamples
from coordination.module.observation.serial_gaussian_observation import \
    SerialGaussianObservation
from coordination.module.transformation.mlp import MLP
from coordination.inference.inference_data import InferenceData
from copy import deepcopy
from coordination.common.constants import DEFAULT_SEED
from coordination.inference.inference_data import InferenceData
from coordination.metadata.serial import SerialMetadata
from coordination.common.utils import adjust_dimensions
from coordination.module.coordination.constant_coordination import ConstantCoordination
from coordination.common.functions import logit
import logging
from coordination.module.latent_component.serial_2d_gaussian_latent_component import \
    Serial2DGaussianLatentComponent
from coordination.module.module import Module
from coordination.module.transformation.sequential import Sequential
from coordination.module.transformation.dimension_reduction import \
    DimensionReduction


class VocalicModel(ModelTemplate):
    """
    This class represents a vocalic model where subjects are talking to each other and their
    speech vocalics are observed as they finish talking.
    """

    def __init__(
            self, config_bundle: VocalicConfigBundle, pymc_model: Optional[pm.Model] = None
    ):
        """
        Creates a vocalic model.

        @param config_bundle: container for the different parameters of the vocalic model.
        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        """
        super().__init__(pymc_model=pymc_model, config_bundle=config_bundle)

    def _register_metadata(self, config_bundle: VocalicConfigBundle):
        """
        Add entries to the metadata dictionary from values filled in a config bundle. This will
        allow adjustment of time steps later if we want to fit/sample less time steps than the
        informed in the original config bundle.
        """
        if "speech_vocalics" in self.metadata:
            metadata: SerialMetadata = self.metadata["speech_vocalics"]
            metadata.num_subjects = config_bundle.num_subjects
            metadata.time_steps_in_coordination_scale = (
                config_bundle.time_steps_in_coordination_scale)
            metadata.subject_indices = config_bundle.subject_indices
            metadata.prev_time_same_subject = config_bundle.prev_time_same_subject
            metadata.prev_time_diff_subject = config_bundle.prev_time_diff_subject
            metadata.observed_values = config_bundle.observed_values
            metadata.normalization_method = config_bundle.observation_normalization
        else:
            self.metadata["speech_vocalics"] = SerialMetadata(
                num_subjects=config_bundle.num_subjects,
                time_steps_in_coordination_scale=config_bundle.time_steps_in_coordination_scale,
                subject_indices=config_bundle.subject_indices,
                prev_time_same_subject=config_bundle.prev_time_same_subject,
                prev_time_diff_subject=config_bundle.prev_time_diff_subject,
                observed_values=config_bundle.observed_values,
                normalization_method=config_bundle.observation_normalization
            )

    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """

        bundle = self._get_adjusted_bundle()

        if bundle.constant_coordination:
            coordination = ConstantCoordination(
                pymc_model=self.pymc_model,
                num_time_steps=bundle.num_time_steps_to_fit,
                alpha_c=bundle.alpha_c,
                beta_c=bundle.beta_c,
                initial_samples=bundle.initial_coordination_samples,
                observed_value=bundle.observed_coordination_for_inference
            )
        else:
            given_coordination = None
            if bundle.observed_coordination_for_inference is not None:
                given_coordination = adjust_dimensions(
                    logit(bundle.observed_coordination_for_inference),
                    bundle.num_time_steps_to_fit
                )

            initial_samples = None
            if bundle.initial_coordination_samples is not None:
                initial_samples = logit(bundle.initial_coordination_samples)
            coordination = SigmoidGaussianCoordination(
                pymc_model=self.pymc_model,
                num_time_steps=bundle.num_time_steps_to_fit,
                mean_mean_uc0=bundle.mean_mean_uc0,
                sd_mean_uc0=bundle.sd_mean_uc0,
                sd_sd_uc=bundle.sd_sd_uc,
                mean_uc0=bundle.mean_uc0,
                sd_uc=bundle.sd_uc,
                initial_samples=initial_samples,
                unbounded_coordination_observed_values=given_coordination
            )

        if bundle.state_space_2d:
            groups = self._create_vocalic_2d_groups(bundle)
        else:
            groups = self._create_vocalic_groups(bundle)

        self._model = Model(
            name="vocalic_model",
            pymc_model=self.pymc_model,
            coordination=coordination,
            component_groups=groups
        )

    def _get_adjusted_bundle(self) -> VocalicConfigBundle:
        """
        Gets a config bundle with time scale adjusted to the scale matching and fitting options.

        @return: adjusted bundle to use in the construction of the modules.
        """
        bundle = self.config_bundle
        if self.config_bundle.match_vocalic_scale:
            if self.config_bundle.time_steps_in_coordination_scale is not None:
                # We estimate coordination at only when at the time steps we have observations.
                # So, we adjust the number of time steps in the coordination scale to match the
                # number of time steps in the vocalic component scale
                bundle = deepcopy(self.config_bundle)

                # We adjust the number of time steps in coordination scale to match that.
                bundle.num_time_steps_in_coordination_scale = len(
                    self.config_bundle.time_steps_in_coordination_scale)

                # Now the time steps of the vocalics won't have any gaps. They will be 0,1,2,...,n,
                # where n is the number of observations.
                bundle.time_steps_in_coordination_scale = np.arange(
                    len(self.config_bundle.time_steps_in_coordination_scale))

        return self.new_config_bundle_from_time_step_info(bundle)

    def _create_vocalic_2d_groups(self, bundle: VocalicConfigBundle) -> List[ComponentGroup]:
        """
        Creates component groups for a model of vocalics with 2D state space.

        @param bundle: config bundle holding information on how to parameterize the modules.
        @return: a list of component groups to be added to the model.
        """
        vocalic_metadata: SerialMetadata = self.metadata["speech_vocalics"]

        # In the 2D case, it may be interesting having multiple state space chains with their
        # own dynamics if different features of a modality have different movement dynamics.
        vocalic_groups = bundle.vocalic_groups
        if vocalic_groups is None:
            vocalic_groups = [
                {
                    "name": None,
                    "features": bundle.vocalic_feature_names,
                    "weights": bundle.weights,
                }
            ]

        groups = []
        for vocalic_group in vocalic_groups:
            # Form a tensor of observations by getting only the dimensions of the features
            # in the group.
            feature_idx = [
                bundle.vocalic_feature_names.index(feature)
                for feature in vocalic_group["features"]
            ]

            observed_values = np.take_along_axis(
                vocalic_metadata.normalized_observations,
                indices=np.array(feature_idx, dtype=int)[:, None],
                axis=0,
            ) if bundle.observed_values is not None else None

            # For retro-compatibility, we only add suffix if groups were defined.

            group_name = vocalic_group["name"]
            suffix = (
                "" if bundle.vocalic_groups is None else f"_{group_name}"
            )

            state_space = Serial2DGaussianLatentComponent(
                uuid=f"state_space{suffix}",
                pymc_model=self.pymc_model,
                num_subjects=bundle.num_subjects,
                mean_mean_a0=bundle.mean_mean_a0,
                sd_mean_a0=bundle.sd_mean_a0,
                sd_sd_a=bundle.sd_sd_a,
                share_mean_a0_across_subjects=bundle.share_mean_a0_across_subjects,
                share_sd_a_across_subjects=bundle.share_sd_a_across_subjects,
                share_mean_a0_across_dimensions=bundle.share_mean_a0_across_dimensions,
                share_sd_a_across_dimensions=bundle.share_sd_a_across_dimensions,
                sampling_time_scale_density=bundle.sampling_time_scale_density,
                allow_sampled_subject_repetition=bundle.allow_sampled_subject_repetition,
                fix_sampled_subject_sequence=bundle.fix_sampled_subject_sequence,
                time_steps_in_coordination_scale=(
                    vocalic_metadata.time_steps_in_coordination_scale
                ),
                prev_time_same_subject=vocalic_metadata.prev_time_same_subject,
                prev_time_diff_subject=vocalic_metadata.prev_time_diff_subject,
                subject_indices=vocalic_metadata.subject_indices,
                mean_a0=bundle.mean_a0,
                sd_a=bundle.sd_a,
                initial_samples=bundle.initial_state_space_samples
            )

            transformation = Sequential(
                child_transformations=[
                    DimensionReduction(keep_dimensions=[0], axis=0),  # position,
                    MLP(
                        uuid=f"state_space_to_speech_vocalics_mlp{suffix}",
                        pymc_model=self.pymc_model,
                        output_dimension_size=len(vocalic_group["features"]),
                        mean_w0=bundle.mean_w0,
                        sd_w0=bundle.sd_w0,
                        num_hidden_layers=bundle.num_hidden_layers,
                        hidden_dimension_size=bundle.hidden_dimension_size,
                        activation=bundle.activation,
                        axis=0,  # Vocalic features axis
                        weights=vocalic_group["weights"],
                    ),
                ]
            )

            observation = SerialGaussianObservation(
                uuid=f"speech_vocalics{suffix}",
                pymc_model=self.pymc_model,
                num_subjects=bundle.num_subjects,
                dimension_size=len(vocalic_group["features"]),
                sd_sd_o=bundle.sd_sd_o,
                share_sd_o_across_subjects=bundle.share_sd_o_across_subjects,
                share_sd_o_across_dimensions=bundle.share_sd_o_across_dimensions,
                dimension_names=vocalic_group["features"],
                observed_values=observed_values,
                time_steps_in_coordination_scale=(
                    vocalic_metadata.time_steps_in_coordination_scale
                ),
                subject_indices=vocalic_metadata.subject_indices,
                sd_o=bundle.sd_o,
            )

            group = ComponentGroup(
                uuid=f"group{suffix}",
                pymc_model=self.pymc_model,
                latent_component=state_space,
                observations=[observation],
                transformations=[transformation],
            )
            groups.append(group)

        return groups

    def _create_vocalic_groups(self, bundle: VocalicConfigBundle) -> List[ComponentGroup]:
        """
        Creates component groups for a model of vocalics.

        @param bundle: config bundle holding information on how to parameterize the modules.
        @return: a list of component groups to be added to the model.
        """
        vocalic_metadata: SerialMetadata = self.metadata["speech_vocalics"]
        state_space = SerialGaussianLatentComponent(
            uuid=f"state_space",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=bundle.state_space_dimension_size,
            self_dependent=True,
            mean_mean_a0=bundle.mean_mean_a0,
            sd_mean_a0=bundle.sd_mean_a0,
            sd_sd_a=bundle.sd_sd_a,
            share_mean_a0_across_subjects=bundle.share_mean_a0_across_subjects,
            share_sd_a_across_subjects=bundle.share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=bundle.share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=bundle.share_sd_a_across_dimensions,
            dimension_names=bundle.state_space_dimension_names,
            sampling_time_scale_density=bundle.sampling_time_scale_density,
            allow_sampled_subject_repetition=bundle.allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=bundle.fix_sampled_subject_sequence,
            time_steps_in_coordination_scale=(
                vocalic_metadata.time_steps_in_coordination_scale
            ),
            prev_time_same_subject=vocalic_metadata.prev_time_same_subject,
            prev_time_diff_subject=vocalic_metadata.prev_time_diff_subject,
            subject_indices=vocalic_metadata.subject_indices,
            mean_a0=bundle.mean_a0,
            sd_a=bundle.sd_a,
            initial_samples=bundle.initial_state_space_samples
        )

        observation = SerialGaussianObservation(
            uuid="speech_vocalics",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=bundle.num_vocalic_features,
            sd_sd_o=bundle.sd_sd_o,
            share_sd_o_across_subjects=bundle.share_sd_o_across_subjects,
            share_sd_o_across_dimensions=bundle.share_sd_o_across_dimensions,
            dimension_names=bundle.vocalic_feature_names,
            observed_values=vocalic_metadata.normalized_observations,
            time_steps_in_coordination_scale=(
                vocalic_metadata.time_steps_in_coordination_scale
            ),
            subject_indices=vocalic_metadata.subject_indices,
            sd_o=bundle.sd_o,
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=None,
        )

        return [group]

    def new_config_bundle_from_posterior_samples(
            self,
            config_bundle: VocalicConfigBundle,
            idata: InferenceData,
            num_samples: int,
            seed: int = DEFAULT_SEED) -> VocalicConfigBundle:
        """
        Uses samples from posterior to update a config bundle. Here we set the samples from the
        posterior in the last time step as initial values for the latent variables. This
        allows us to generate samples in the future for predictive checks.

        @param config_bundle: original config bundle.
        @param idata: inference data.
        @param num_samples: number of samples from posterior to use. Samples will be chosen
            randomly from the posterior samples.
        @param seed: random seed for reproducibility when choosing the samples to keep.
        """
        new_bundle = deepcopy(config_bundle)

        np.random.seed(seed)
        samples_idx = np.random.choice(idata.num_posterior_samples, num_samples, replace=False)

        new_bundle.mean_a0 = idata.get_posterior_samples("state_space_mean_a0", samples_idx)
        new_bundle.sd_a = idata.get_posterior_samples("state_space_sd_a", samples_idx)
        new_bundle.sd_o = idata.get_posterior_samples("speech_vocalics_sd_o", samples_idx)

        if config_bundle.constant_coordination:
            new_bundle.initial_coordination_samples = (idata.get_posterior_samples(
                "coordination", samples_idx))
        else:
            new_bundle.mean_uc0 = idata.get_posterior_samples("coordination_mean_uc0")[
                samples_idx]
            new_bundle.sd_uc = idata.get_posterior_samples("coordination_sd_uc", samples_idx)
            new_bundle.initial_coordination_samples = (
                idata.get_posterior_samples("coordination", samples_idx))

        new_bundle.initial_state_space_samples = (
            idata.get_posterior_samples("state_space", samples_idx))

        return new_bundle
