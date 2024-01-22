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
    SerialGaussianLatentComponent
from coordination.module.observation.serial_gaussian_observation import \
    SerialGaussianObservation
from coordination.module.transformation.mlp import MLP
from coordination.inference.inference_data import InferenceData
from copy import deepcopy


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

        if (self.config_bundle.match_vocalics_scale and
                self.config_bundle.time_steps_in_coordination_scale is not None):
            c_num_time_steps = len(self.config_bundle.time_steps_in_coordination_scale)
        else:
            c_num_time_steps = self.config_bundle.num_time_steps_in_coordination_scale

        c_num_time_steps = int(c_num_time_steps * self.config_bundle.p_time_steps_to_fit)

        coordination = SigmoidGaussianCoordination(
            pymc_model=self.pymc_model,
            num_time_steps=c_num_time_steps,
            mean_mean_uc0=self.config_bundle.mean_mean_uc0,
            sd_mean_uc0=self.config_bundle.sd_mean_uc0,
            sd_sd_uc=self.config_bundle.sd_sd_uc,
            mean_uc0=self.config_bundle.mean_uc0,
            sd_uc=self.config_bundle.sd_uc,
        )

        # In case we are fitting less number of time steps, adjust indices in the component.
        time_steps = self.config_bundle.time_steps_in_coordination_scale
        if time_steps is not None:
            time_steps = time_steps[time_steps < c_num_time_steps]

        prev_time_same_subject = self.config_bundle.prev_time_same_subject
        if prev_time_same_subject is not None:
            prev_time_same_subject = prev_time_same_subject[:len(time_steps)]

        prev_time_diff_subject = self.config_bundle.prev_time_diff_subject
        if prev_time_diff_subject is not None:
            prev_time_diff_subject = prev_time_diff_subject[:len(time_steps)]

        subject_indices = self.config_bundle.subject_indices
        if subject_indices is not None:
            subject_indices = subject_indices[:len(time_steps)]

        state_space = SerialGaussianLatentComponent(
            uuid="state_space",
            pymc_model=self.pymc_model,
            num_subjects=self.config_bundle.num_subjects,
            dimension_size=self.config_bundle.state_space_dimension_size,
            self_dependent=self.config_bundle.self_dependent,
            mean_mean_a0=self.config_bundle.mean_mean_a0,
            sd_mean_a0=self.config_bundle.sd_mean_a0,
            sd_sd_a=self.config_bundle.sd_sd_a,
            share_mean_a0_across_subjects=self.config_bundle.share_mean_a0_across_subjects,
            share_sd_a_across_subjects=self.config_bundle.share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=self.config_bundle.share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=self.config_bundle.share_sd_a_across_dimensions,
            dimension_names=self.config_bundle.state_space_dimension_names,
            sampling_time_scale_density=self.config_bundle.sampling_time_scale_density,
            allow_sampled_subject_repetition=self.config_bundle.allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=self.config_bundle.fix_sampled_subject_sequence,
            time_steps_in_coordination_scale=time_steps,
            prev_time_same_subject=prev_time_same_subject,
            prev_time_diff_subject=prev_time_diff_subject,
            subject_indices=subject_indices,
            mean_a0=self.config_bundle.mean_a0,
            sd_a=self.config_bundle.sd_a,
        )

        transformation = None
        if self.config_bundle.activation != "linear" or (
                self.config_bundle.state_space_dimension_size
                < self.config_bundle.num_vocalic_features
        ):
            # Transform latent samples before passing to the observation module to account for
            # non-linearity and/or different dimensions between the latent component and
            # associated observation
            transformation = MLP(
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
            )

        observed_values = self.config_bundle.observed_values
        if observed_values is not None:
            observed_values = observed_values[..., :len(time_steps)]
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
            observed_values=observed_values,
            time_steps_in_coordination_scale=time_steps,
            subject_indices=subject_indices,
            sd_o=self.config_bundle.sd_o,
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
            coordination_samples=self.config_bundle.coordination_samples,
        )

    def update_config_bundle_from_posterior_samples(self,
                                                    idata: InferenceData) -> VocalicConfigBundle:
        """
        Uses samples from posterior to update a config bundle. Here we set the samples from the
        posterior in the last time step as initial values for the latent variables. This
        allows us to generate samples in the future for predictive checks.

        @param idata: inference data.
        """
        upd_bundle = deepcopy(self.config_bundle)

        # Samples are in the first dimension

        upd_bundle.mean_uc0 = (
            idata.trace.posterior["unbounded_coordination"].stack(
                sample=["draw", "chain"]).to_numpy()[-1])

        subject_indices = np.array(
            [
                int(x.split("#")[0])
                for x in getattr(means, f"state_space_time").data
            ]
        )


        upd_bundle.mean_a0 = (
            idata.trace.posterior["state_space"].stack(
                sample=["draw", "chain"]).to_numpy()[:, -1, :][:, None]).T

        if "coordination_sd_uc" in idata.trace.observed_data:
            upd_bundle.sd_uc = idata.trace.observed_data["coordination_sd_uc"].to_numpy()
        else:
            upd_bundle.sd_uc = (
                                   idata.trace.posterior["coordination_sd_uc"].stack(
                                       sample=["draw", "chain"]).to_numpy())[:, None]

        if "state_space_sd_a" in idata.trace.observed_data:
            upd_bundle.sd_a = idata.trace.observed_data["state_space_sd_a"].to_numpy()
        else:
            upd_bundle.sd_a = (
                idata.trace.posterior["state_space_sd_a"].stack(
                    sample=["draw", "chain"]).to_numpy())
            upd_bundle.sd_a = np.moveaxis(upd_bundle.sd_a, -1, 0)

        if "speech_vocalics_sd_o" in idata.trace.observed_data:
            upd_bundle.sd_o = idata.trace.observed_data["speech_vocalics_sd_o"].to_numpy()
        else:
            upd_bundle.sd_o = (
                idata.trace.posterior["speech_vocalics_sd_o"].stack(
                    sample=["draw", "chain"]).to_numpy())
            upd_bundle.sd_o = np.moveaxis(upd_bundle.sd_o, -1, 0)

        return upd_bundle
