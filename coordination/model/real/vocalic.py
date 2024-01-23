from typing import Optional

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
        super().__init__(pymc_model, config_bundle)

    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """

        bundle = VocalicModel.new_config_bundle_from_time_step_info(self.config_bundle)

        coordination = SigmoidGaussianCoordination(
            pymc_model=self.pymc_model,
            num_time_steps=bundle.num_time_steps_in_coordination_scale,
            mean_mean_uc0=bundle.mean_mean_uc0,
            sd_mean_uc0=bundle.sd_mean_uc0,
            sd_sd_uc=bundle.sd_sd_uc,
            mean_uc0=bundle.mean_uc0,
            sd_uc=bundle.sd_uc,
            posterior_samples=bundle.unbounded_coordination_posterior_samples
        )

        state_space = SerialGaussianLatentComponent(
            uuid="state_space",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=bundle.state_space_dimension_size,
            self_dependent=bundle.self_dependent,
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
            time_steps_in_coordination_scale=bundle.time_steps_in_coordination_scale,
            prev_time_same_subject=bundle.prev_time_same_subject,
            prev_time_diff_subject=bundle.prev_time_diff_subject,
            subject_indices=bundle.subject_indices,
            mean_a0=bundle.mean_a0,
            sd_a=bundle.sd_a,
            posterior_samples=bundle.state_space_posterior_samples
        )

        transformation = None
        if bundle.activation != "linear" or (
                bundle.state_space_dimension_size
                < bundle.num_vocalic_features
        ):
            # Transform latent samples before passing to the observation module to account for
            # non-linearity and/or different dimensions between the latent component and
            # associated observation
            transformation = MLP(
                uuid="state_space_to_speech_vocalics_mlp",
                pymc_model=self.pymc_model,
                output_dimension_size=bundle.num_vocalic_features,
                mean_w0=bundle.mean_w0,
                sd_w0=bundle.sd_w0,
                num_hidden_layers=bundle.num_hidden_layers,
                hidden_dimension_size=bundle.hidden_dimension_size,
                activation=bundle.activation,
                axis=0,  # Vocalic features axis
                weights=bundle.weights,
            )

        observation = SerialGaussianObservation(
            uuid="speech_vocalics",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=bundle.num_vocalic_features,
            sd_sd_o=bundle.sd_sd_o,
            share_sd_o_across_subjects=bundle.share_sd_o_across_subjects,
            share_sd_o_across_dimensions=bundle.share_sd_o_across_dimensions,
            normalization=bundle.observation_normalization,
            dimension_names=bundle.vocalic_feature_names,
            observed_values=bundle.observed_values,
            time_steps_in_coordination_scale=bundle.time_steps_in_coordination_scale,
            subject_indices=bundle.subject_indices,
            sd_o=bundle.sd_o,
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=[transformation] if transformation else None,
        )

        self._model = Model(
            name="vocalic_model",
            pymc_model=self.pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=bundle.coordination_samples,
        )

    @staticmethod
    def new_config_bundle_from_time_step_info(
            config_bundle: VocalicConfigBundle) -> VocalicConfigBundle:
        """
        Gets a new config bundle with metadata and observed values adapted to the number of time
        steps in coordination scale in case we don't want to fit just a portion of the time series.

        @param config_bundle: original config bundle.
        @return: new config bundle.
        """
        new_bundle = deepcopy(config_bundle)

        if (config_bundle.match_vocalics_scale and
                config_bundle.time_steps_in_coordination_scale is not None):
            new_bundle.num_time_steps_in_coordination_scale = len(
                config_bundle.time_steps_in_coordination_scale)
            new_bundle.time_steps_in_coordination_scale = np.arange(
                new_bundle.num_time_steps_in_coordination_scale)

        new_bundle.num_time_steps_in_coordination_scale = int(
            new_bundle.num_time_steps_in_coordination_scale * config_bundle.p_time_steps_to_fit)

        # State space info
        # In case we are fitting less number of time steps, adjust indices in the component.
        if new_bundle.time_steps_in_coordination_scale is not None:
            T = new_bundle.num_time_steps_in_coordination_scale
            ts = new_bundle.time_steps_in_coordination_scale
            ts = ts[ts < T]

            new_bundle.time_steps_in_coordination_scale = ts

            if new_bundle.prev_time_same_subject is not None:
                new_bundle.prev_time_same_subject = new_bundle.prev_time_same_subject[:len(ts)]

            if new_bundle.prev_time_diff_subject is not None:
                new_bundle.prev_time_diff_subject = new_bundle.prev_time_diff_subject[:len(ts)]

            if new_bundle.subject_indices is not None:
                new_bundle.subject_indices = new_bundle.subject_indices[:len(ts)]

            if new_bundle.observed_values is not None:
                new_bundle.observed_values = new_bundle.observed_values[..., :len(ts)]

        return new_bundle

    @staticmethod
    def new_config_bundle_from_posterior_samples(config_bundle: VocalicConfigBundle,
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

        new_bundle.mean_uc0 = idata.get_posterior_samples("coordination_mean_uc0")[
            samples_idx]
        new_bundle.sd_uc = idata.get_posterior_samples("coordination_sd_uc", samples_idx)
        new_bundle.mean_a0 = idata.get_posterior_samples("state_space_mean_a0", samples_idx)
        new_bundle.sd_a = idata.get_posterior_samples("state_space_sd_a", samples_idx)
        new_bundle.sd_o = idata.get_posterior_samples("speech_vocalics_sd_o", samples_idx)

        new_bundle.unbounded_coordination_posterior_samples = (
            idata.get_posterior_samples("unbounded_coordination", samples_idx))
        new_bundle.state_space_posterior_samples = (
            idata.get_posterior_samples("state_space", samples_idx))

        return new_bundle
