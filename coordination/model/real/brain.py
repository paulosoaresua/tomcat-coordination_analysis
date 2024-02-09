from copy import deepcopy
from typing import List, Optional

import numpy as np
import pymc as pm

from coordination.common.constants import DEFAULT_SEED
from coordination.common.functions import logit
from coordination.common.utils import adjust_dimensions
from coordination.inference.inference_data import InferenceData
from coordination.metadata.non_serial import NonSerialMetadata
from coordination.model.config_bundle.brain import BrainBundle
from coordination.model.model import Model
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.constant_coordination import \
    ConstantCoordination
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.non_serial_2d_gaussian_latent_component import \
    NonSerial2DGaussianLatentComponent
from coordination.module.observation.non_serial_gaussian_observation import \
    NonSerialGaussianObservation
from coordination.module.transformation.dimension_reduction import \
    DimensionReduction
from coordination.module.transformation.mlp import MLP
from coordination.module.transformation.sequential import Sequential


class BrainModel(ModelTemplate):
    """
    This class represents a brain model that measures how coordination affects the entrainment
    among fNIRS and EKG signals across different subjects.
    """

    def __init__(
            self, config_bundle: BrainBundle, pymc_model: Optional[pm.Model] = None
    ):
        """
        Creates a brain model.

        @param config_bundle: container for the different parameters of the brain model.
        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        """
        super().__init__(pymc_model=pymc_model, config_bundle=config_bundle)

    def _register_metadata(self, config_bundle: BrainBundle):
        """
        Add entries to the metadata dictionary from values filled in a config bundle. This will
        allow adjustment of time steps later if we want to fit/sample less time steps than the
        informed in the original config bundle.
        """
        if config_bundle.fnirs_groups is None:
            fnirs_groups = [{"name": None, "features": config_bundle.fnirs_channel_names}]
        else:
            fnirs_groups = config_bundle.fnirs_groups

        for fnirs_group in fnirs_groups:
            # Form a tensor of observations by getting only the dimensions of the features
            # in the group.
            feature_idx = [
                config_bundle.fnirs_channel_names.index(feature)
                for feature in fnirs_group["features"]
            ]

            observed_values = (
                np.take_along_axis(
                    config_bundle.fnirs_observed_values,
                    indices=np.array(feature_idx, dtype=int)[None, :, None],
                    axis=1,
                )
                if config_bundle.fnirs_observed_values is not None
                else None
            )

            # For retro-compatibility, we only add suffix if groups were defined.
            group_name = fnirs_group["name"]
            suffix = "" if config_bundle.fnirs_groups is None else f"_{group_name}"

            if f"fnirs{suffix}" in self.metadata:
                metadata: NonSerialMetadata = self.metadata[f"fnirs{suffix}"]
                metadata.time_steps_in_coordination_scale = (
                    config_bundle.fnirs_time_steps_in_coordination_scale
                )
                metadata.observed_values = observed_values
                metadata.normalization_method = config_bundle.observation_normalization
            else:
                self.metadata[f"fnirs{suffix}"] = NonSerialMetadata(
                    time_steps_in_coordination_scale=(
                        config_bundle.fnirs_time_steps_in_coordination_scale
                    ),
                    observed_values=observed_values,
                    normalization_method=config_bundle.observation_normalization,
                )

        if config_bundle.include_gsr:
            if "gsr" in self.metadata:
                metadata: NonSerialMetadata = self.metadata["gsr"]
                metadata.time_steps_in_coordination_scale = (
                    config_bundle.gsr_time_steps_in_coordination_scale
                )
                metadata.observed_values = config_bundle.gsr_observed_values
                metadata.normalization_method = config_bundle.observation_normalization
            else:
                self.metadata["gsr"] = NonSerialMetadata(
                    time_steps_in_coordination_scale=(
                        config_bundle.gsr_time_steps_in_coordination_scale
                    ),
                    observed_values=config_bundle.gsr_observed_values,
                    normalization_method=config_bundle.observation_normalization,
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
                observed_value=bundle.observed_coordination_for_inference,
            )
        else:
            given_coordination = None
            if bundle.observed_coordination_for_inference is not None:
                given_coordination = adjust_dimensions(
                    logit(bundle.observed_coordination_for_inference),
                    bundle.num_time_steps_to_fit,
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
                unbounded_coordination_observed_values=given_coordination,
            )

        groups = self._create_fnirs_groups(bundle)

        if bundle.include_gsr:
            groups.append(self._create_gsr_group(bundle))

        name = "brain_fnirs"
        if bundle.include_gsr:
            name += "_gsr"

        self._model = Model(
            name=f"{name}_model",
            pymc_model=self.pymc_model,
            coordination=coordination,
            component_groups=groups,
        )

    def _get_adjusted_bundle(self) -> BrainBundle:
        """
        Gets a config bundle with time scale adjusted to the scale matching and fitting options.

        @return: adjusted bundle to use in the construction of the modules.
        """
        return self.new_config_bundle_from_time_step_info(self.config_bundle)

    def _create_fnirs_groups(self, bundle: BrainBundle) -> List[ComponentGroup]:
        """
        Creates component groups for the fnirs component.

        @param bundle: config bundle holding information on how to parameterize the modules.
        @return: a list of component groups to be added to the model.
        """
        # In the 2D case, it may be interesting having multiple state space chains with their
        # own dynamics if different features of a modality have different movement dynamics.
        fnirs_groups = bundle.fnirs_groups
        if fnirs_groups is None:
            fnirs_groups = [{"name": None, "features": bundle.fnirs_channel_names,
                             "asymmetric_coordination": bundle.asymmetric_coordination}]

        groups = []
        for i, fnirs_group in enumerate(fnirs_groups):
            # For retro-compatibility, we only add suffix if groups were defined.
            group_name = fnirs_group["name"]
            suffix = "" if bundle.fnirs_groups is None else f"_{group_name}"

            fnirs_metadata = self.metadata[f"fnirs{suffix}"]

            # One value for each group can be given in form of a list. This also happens when
            # filling these parameters from the posterior samples from different groups.
            if isinstance(bundle.fnirs_mean_a0, list):
                mean_a0 = bundle.fnirs_mean_a0[i]
            else:
                mean_a0 = bundle.fnirs_mean_a0

            if isinstance(bundle.fnirs_sd_a, list):
                sd_a = bundle.fnirs_sd_a[i]
            else:
                sd_a = bundle.fnirs_sd_a

            if isinstance(bundle.initial_fnirs_state_space_samples, list):
                initial_state_space_samples = bundle.initial_fnirs_state_space_samples[i]
            else:
                initial_state_space_samples = bundle.initial_fnirs_state_space_samples

            state_space = NonSerial2DGaussianLatentComponent(
                uuid=f"fnirs_state_space{suffix}",
                pymc_model=self.pymc_model,
                num_subjects=bundle.num_subjects,
                mean_mean_a0=bundle.fnirs_mean_mean_a0,
                sd_mean_a0=bundle.fnirs_sd_mean_a0,
                sd_sd_a=bundle.fnirs_sd_sd_a,
                share_mean_a0_across_subjects=bundle.fnirs_share_mean_a0_across_subjects,
                share_sd_a_across_subjects=bundle.fnirs_share_sd_a_across_subjects,
                share_mean_a0_across_dimensions=bundle.fnirs_share_mean_a0_across_dimensions,
                share_sd_a_across_dimensions=bundle.fnirs_share_sd_a_across_dimensions,
                time_steps_in_coordination_scale=(
                    fnirs_metadata.time_steps_in_coordination_scale
                ),
                mean_a0=mean_a0,
                sd_a=sd_a,
                sampling_relative_frequency=bundle.sampling_relative_frequency,
                initial_samples=initial_state_space_samples,
                asymmetric_coordination=fnirs_group["asymmetric_coordination"]
            )
            # We assume data is normalized and add a transformation with fixed unitary weights that
            # bring the position in the state space to a collection of channels in the
            # observations.
            transformation = Sequential(
                child_transformations=[
                    DimensionReduction(keep_dimensions=[0], axis=1),  # position,
                    MLP(
                        uuid=f"fnirs_state_space_to_speech_vocalics_mlp{suffix}",
                        pymc_model=self.pymc_model,
                        output_dimension_size=len(fnirs_group["features"]),
                        mean_w0=0,
                        sd_w0=1,
                        num_hidden_layers=0,
                        hidden_dimension_size=0,
                        activation="linear",
                        axis=1,  # fnirs channel axis
                        weights=[np.ones((1, len(fnirs_group["features"])))],
                    ),
                ]
            )

            if isinstance(bundle.fnirs_sd_o, list):
                sd_o = bundle.fnirs_sd_o[i]
            else:
                sd_o = bundle.fnirs_sd_o

            observation = NonSerialGaussianObservation(
                uuid=f"fnirs{suffix}",
                pymc_model=self.pymc_model,
                num_subjects=bundle.num_subjects,
                dimension_size=len(fnirs_group["features"]),
                sd_sd_o=bundle.fnirs_sd_sd_o,
                share_sd_o_across_subjects=bundle.fnirs_share_sd_o_across_subjects,
                share_sd_o_across_dimensions=bundle.fnirs_share_sd_o_across_dimensions,
                dimension_names=fnirs_group["features"],
                observed_values=fnirs_metadata.normalized_observations,
                time_steps_in_coordination_scale=(
                    fnirs_metadata.time_steps_in_coordination_scale
                ),
                sd_o=sd_o,
            )

            group = ComponentGroup(
                uuid=f"fnirs_group{suffix}",
                pymc_model=self.pymc_model,
                latent_component=state_space,
                observations=[observation],
                transformations=[transformation],
            )
            groups.append(group)

        return groups

    def _create_gsr_group(self, bundle: BrainBundle) -> ComponentGroup:
        """
        Creates component groups for the gsr component.

        @param bundle: config bundle holding information on how to parameterize the modules.
        @return: a list of component groups to be added to the model.
        """
        # In the 2D case, it may be interesting having multiple state space chains with their
        # own dynamics if different features of a modality have different movement dynamics.
        gsr_metadata = self.metadata["gsr"]

        # One value for each group can be given in form of a list. This also happens when
        # filling these parameters from the posterior samples from different groups.
        state_space = NonSerial2DGaussianLatentComponent(
            uuid=f"gsr_state_space",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            mean_mean_a0=bundle.gsr_mean_mean_a0,
            sd_mean_a0=bundle.gsr_sd_mean_a0,
            sd_sd_a=bundle.gsr_sd_sd_a,
            share_mean_a0_across_subjects=bundle.gsr_share_mean_a0_across_subjects,
            share_sd_a_across_subjects=bundle.gsr_share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=bundle.gsr_share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=bundle.gsr_share_sd_a_across_dimensions,
            time_steps_in_coordination_scale=(
                gsr_metadata.time_steps_in_coordination_scale
            ),
            mean_a0=bundle.gsr_mean_a0,
            sd_a=bundle.gsr_sd_a,
            sampling_relative_frequency=bundle.sampling_relative_frequency,
            initial_samples=bundle.initial_gsr_state_space_samples,
            asymmetric_coordination=bundle.gsr_asymmetric_coordination
        )
        # We assume data is normalized and add a transformation with fixed unitary weights that
        # bring the position in the state space to a collection of channels in the
        # observations.
        transformation = Sequential(
            child_transformations=[
                DimensionReduction(keep_dimensions=[0], axis=1),  # position,
                MLP(
                    uuid="gsr_state_space_to_speech_vocalics_mlp",
                    pymc_model=self.pymc_model,
                    output_dimension_size=1,
                    mean_w0=0,
                    sd_w0=1,
                    num_hidden_layers=0,
                    hidden_dimension_size=0,
                    activation="linear",
                    axis=1,  # gsr feature axis
                    weights=[np.ones((1, 1))],
                ),
            ]
        )

        observation = NonSerialGaussianObservation(
            uuid="gsr",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=1,
            sd_sd_o=bundle.gsr_sd_sd_o,
            share_sd_o_across_subjects=bundle.gsr_share_sd_o_across_subjects,
            share_sd_o_across_dimensions=True,
            dimension_names=["gsr"],
            observed_values=gsr_metadata.normalized_observations,
            time_steps_in_coordination_scale=(
                gsr_metadata.time_steps_in_coordination_scale
            ),
            sd_o=bundle.gsr_sd_o,
        )

        group = ComponentGroup(
            uuid=f"gsr_group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=[transformation],
        )

        return group

    def new_config_bundle_from_posterior_samples(
            self,
            config_bundle: BrainBundle,
            idata: InferenceData,
            num_samples: int,
            seed: int = DEFAULT_SEED,
    ) -> BrainBundle:
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
        samples_idx = np.random.choice(
            idata.num_posterior_samples, num_samples, replace=False
        )

        fnirs_groups = new_bundle.fnirs_groups
        if fnirs_groups is None:
            fnirs_groups = [{"name": None, "features": new_bundle.fnirs_channel_names}]

        new_bundle.fnirs_mean_a0 = []
        new_bundle.fnirs_sd_a = []
        new_bundle.fnirs_sd_o = []
        new_bundle.initial_fnirs_state_space_samples = []
        for fnirs_group in fnirs_groups:
            # For retro-compatibility, we only add suffix if groups were defined.
            group_name = fnirs_group["name"]
            suffix = "" if new_bundle.fnirs_groups is None else f"_{group_name}"

            new_bundle.fnirs_mean_a0.append(idata.get_posterior_samples(
                f"fnirs_state_space{suffix}_mean_a0", samples_idx
            ))
            new_bundle.fnirs_sd_a.append(idata.get_posterior_samples(
                f"fnirs_state_space{suffix}_sd_a", samples_idx
            ))
            new_bundle.fnirs_sd_o.append(idata.get_posterior_samples(
                f"fnirs{suffix}_sd_o", samples_idx
            ))
            new_bundle.initial_fnirs_state_space_samples.append(idata.get_posterior_samples(
                f"fnirs_state_space{suffix}", samples_idx
            ))

        new_bundle.gsr_mean_a0 = idata.get_posterior_samples(
                f"gsr_state_space_mean_a0", samples_idx
            )
        new_bundle.gsr_sd_a = idata.get_posterior_samples(
                f"gsr_state_space_sd_a", samples_idx
            )
        new_bundle.gsr_sd_o = idata.get_posterior_samples(
                f"gsr_sd_o", samples_idx
            )
        new_bundle.initial_gsr_state_space_samples = idata.get_posterior_samples(
                f"gsr_state_space", samples_idx
            )

        if config_bundle.constant_coordination:
            new_bundle.initial_coordination_samples = idata.get_posterior_samples(
                "coordination", samples_idx
            )
        else:
            new_bundle.mean_uc0 = idata.get_posterior_samples(
                "coordination_mean_uc0", samples_idx
            )
            new_bundle.sd_uc = idata.get_posterior_samples(
                "coordination_sd_uc", samples_idx
            )
            new_bundle.initial_coordination_samples = idata.get_posterior_samples(
                "coordination", samples_idx
            )

        return new_bundle
