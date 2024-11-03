from copy import deepcopy
from typing import List, Optional

import numpy as np
from coordination.module.coordination.dirichlet_gaussian_coordination_3d import DirichletGaussianCoordination3D
from coordination.module.coordination.sigmoid_gaussian_coordination_3d import SigmoidGaussianCoordination3D
import pymc as pm

from coordination.common.constants import DEFAULT_SEED
from coordination.common.functions import logit
from coordination.common.utils import adjust_dimensions
from coordination.inference.inference_data import InferenceData
from coordination.metadata.non_serial import NonSerialMetadata
from coordination.metadata.serial import SerialMetadata
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
from coordination.module.latent_component.non_serial_gaussian_latent_component import \
    NonSerialGaussianLatentComponent
from coordination.module.latent_component.null_latent_component import \
    NullLatentComponent
from coordination.module.latent_component.serial_gaussian_latent_component import \
    SerialGaussianLatentComponent
from coordination.module.observation.non_serial_gaussian_observation import \
    NonSerialGaussianObservation
from coordination.module.observation.serial_gaussian_observation import \
    SerialGaussianObservation
from coordination.module.observation.spike_observation import SpikeObservation
from coordination.module.transformation.dimension_reduction import \
    DimensionReduction
from coordination.module.transformation.mlp import MLP
from coordination.module.transformation.sequential import Sequential
from coordination.module.common_cause.common_cause_gaussian_2d import CommonCauseGaussian2D


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
            fnirs_groups = [
                {"name": None, "features": config_bundle.fnirs_channel_names}
            ]
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

        if config_bundle.include_vocalic:
            if "speech_vocalics" in self.metadata:
                metadata: SerialMetadata = self.metadata["speech_vocalics"]
                metadata.num_subjects = config_bundle.num_subjects
                metadata.time_steps_in_coordination_scale = (
                    config_bundle.vocalic_time_steps_in_coordination_scale
                )
                metadata.subject_indices = config_bundle.vocalic_subject_indices
                metadata.prev_time_same_subject = (
                    config_bundle.vocalic_prev_time_same_subject
                )
                metadata.prev_time_diff_subject = (
                    config_bundle.vocalic_prev_time_diff_subject
                )
                metadata.observed_values = config_bundle.vocalic_observed_values
                metadata.normalization_method = config_bundle.observation_normalization
            else:
                self.metadata["speech_vocalics"] = SerialMetadata(
                    num_subjects=config_bundle.num_subjects,
                    time_steps_in_coordination_scale=(
                        config_bundle.vocalic_time_steps_in_coordination_scale
                    ),
                    subject_indices=config_bundle.vocalic_subject_indices,
                    prev_time_same_subject=config_bundle.vocalic_prev_time_same_subject,
                    prev_time_diff_subject=config_bundle.vocalic_prev_time_diff_subject,
                    observed_values=config_bundle.vocalic_observed_values,
                    normalization_method=config_bundle.observation_normalization,
                )

        if config_bundle.include_semantic:
            if "semantic_link" in self.metadata:
                metadata: NonSerialMetadata = self.metadata["semantic_link"]
                metadata.time_steps_in_coordination_scale = (
                    config_bundle.semantic_link_time_steps_in_coordination_scale
                )
                if (
                        config_bundle.semantic_link_time_steps_in_coordination_scale
                        is not None
                ):
                    metadata.observed_values = np.ones_like(
                        config_bundle.semantic_link_time_steps_in_coordination_scale
                    )
            else:
                obs = None
                if (
                        config_bundle.semantic_link_time_steps_in_coordination_scale
                        is not None
                ):
                    obs = np.ones_like(
                        config_bundle.semantic_link_time_steps_in_coordination_scale
                    )
                self.metadata["semantic_link"] = NonSerialMetadata(
                    time_steps_in_coordination_scale=(
                        config_bundle.semantic_link_time_steps_in_coordination_scale
                    ),
                    observed_values=obs,
                    normalization_method=None,
                )

    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """

        bundle = self._get_adjusted_bundle()

        coordination = self._create_coordination(bundle)

        groups = self._create_fnirs_groups(bundle)

        if bundle.include_gsr:
            groups.append(self._create_gsr_group(bundle))

        if bundle.include_vocalic:
            groups.append(self._create_vocalic_group(bundle))

        if bundle.include_semantic:
            semantic_link_group = self._create_semantic_group(bundle)
            if semantic_link_group is not None:
                groups.append(semantic_link_group)

        name = "brain_fnirs"
        if bundle.include_gsr:
            name += "_gsr"
        if bundle.include_vocalic:
            name += "_vocalic"
        if bundle.include_semantic:
            name += "_semantic"

        self._model = Model(
            name=f"{name}_model",
            pymc_model=self.pymc_model,
            coordination=coordination,
            component_groups=groups,
        )

    def _create_coordination(self, bundle):
        if bundle.constant_coordination or bundle.fnirs_share_fnirs_latent_state_across_subjects:
            given_coordination = (
                0 if bundle.fnirs_share_fnirs_latent_state_across_subjects else
                bundle.observed_coordination_for_inference)
            coordination = ConstantCoordination(
                pymc_model=self.pymc_model,
                num_time_steps=bundle.num_time_steps_to_fit,
                alpha_c=bundle.alpha_c,
                beta_c=bundle.beta_c,
                initial_samples=bundle.initial_coordination_samples,
                observed_value=given_coordination,
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
            
            # TODO (Ming): Include more parameters in the config bundle to decide which
            #  coordination model to use and parameters to set. For instance,
            #  use_dirichlet_coordination to control if we want to use the dirichlet version of
            #  coordination when common cause = True or the sigmoid + normalization version.
            #  Also include parameters for us to able to set the values of individualism,
            #  coordination and common cause independently when use the newly implemented 3d
            #  coordination modules.
            # ==========================================================================================
            # If common cause is true, we use the dirichlet version of coordination. Otherwise,
            #  we use the sigmoid + normalization version.
            if bundle.common_cause:
                if bundle.coordination_mode == "dirichlet":
                    coordination = DirichletGaussianCoordination3D(
                        pymc_model=self.pymc_model,
                        num_time_steps=bundle.num_time_steps_to_fit,
                        mean_mean_uc0=bundle.mean_mean_uc0,
                        sd_mean_uc0=bundle.sd_mean_uc0,
                        sd_uc=bundle.sd_uc,
                        initial_samples=bundle.initial_coordination_samples,
                        unbounded_coordination_observed_values=bundle.observed_coordination_for_inference,
                    )
                elif bundle.coordination_mode == "sigmoid":
                    coordination = SigmoidGaussianCoordination3D(
                        pymc_model=self.pymc_model,
                        num_time_steps=bundle.num_time_steps_to_fit,
                        mean_mean_uc0=bundle.mean_mean_uc0,
                        sd_mean_uc0=bundle.sd_mean_uc0,
                        sd_uc=bundle.sd_uc,
                        initial_samples=bundle.initial_coordination_samples,
                        unbounded_coordination_observed_values=bundle.observed_coordination_for_inference,
                    )
            else:
                # Use SigmoidGaussianCoordination if common cause is not enabled
                coordination = SigmoidGaussianCoordination(
                    pymc_model=self.pymc_model,
                    num_time_steps=bundle.num_time_steps_to_fit,
                    mean_mean_uc0=bundle.mean_mean_uc0,
                    sd_mean_uc0=bundle.sd_mean_uc0,
                    sd_uc=bundle.sd_uc,
                    initial_samples=bundle.initial_coordination_samples,
                    observed_value=bundle.observed_coordination_for_inference
                )
        # ================================================================================================================
        return coordination

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
            fnirs_groups = [
                {
                    "name": None,
                    "features": bundle.fnirs_channel_names,
                    "asymmetric_coordination": bundle.asymmetric_coordination,
                }
            ]

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
                initial_state_space_samples = bundle.initial_fnirs_state_space_samples[
                    i
                ]
            else:
                initial_state_space_samples = bundle.initial_fnirs_state_space_samples

            if bundle.use_1d_state_space:
                state_space = NonSerialGaussianLatentComponent(
                    uuid=f"fnirs_state_space{suffix}",
                    pymc_model=self.pymc_model,
                    num_subjects=bundle.num_subjects,
                    dimension_size=len(fnirs_group["features"]),
                    dimension_names=[f"latent_{f}" for f in fnirs_group["features"]],
                    self_dependent=bundle.self_dependent_latent_states,
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
                    asymmetric_coordination=fnirs_group.get(
                        "asymmetric_coordination", False
                    ),
                )
                transformation = None
            else:
                state_space = NonSerial2DGaussianLatentComponent(
                    uuid=f"fnirs_state_space{suffix}",
                    pymc_model=self.pymc_model,
                    num_subjects=bundle.num_subjects,
                    self_dependent=bundle.self_dependent_latent_states,
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
                    asymmetric_coordination=fnirs_group.get(
                        "asymmetric_coordination", False
                    ),
                    single_chain=bundle.fnirs_share_fnirs_latent_state_across_subjects,
                    common_cause=bundle.common_cause
                )
                # We assume data is normalized and add a transformation with fixed unitary weights
                # that bring the position in the state space to a collection of channels in the
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

            if bundle.common_cause:
                common_cause_component = CommonCauseGaussian2D(
                    uuid=f"common_cause{suffix}",
                    pymc_model=self.pymc_model,
                    mean_mean_cc0=bundle.mean_mean_cc0,
                    sd_mean_cc0=bundle.sd_mean_cc0,
                    sd_sd_cc=bundle.sd_sd_cc,
                    share_mean_cc0_across_dimensions=bundle.share_mean_cc0_across_dimensions,
                    share_sd_cc_across_dimensions=bundle.share_sd_cc_across_dimensions,
                    time_steps_in_coordination_scale=(
                        fnirs_metadata.time_steps_in_coordination_scale
                    ),
                    mean_cc0=bundle.mean_cc0,
                    sd_cc=bundle.sd_cc,
                    initial_samples=bundle.initial_common_cause_samples,
                )
            else:
                common_cause_component = None

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
                transformations=None if transformation is None else [transformation],
                common_cause=common_cause_component
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
            uuid="gsr_state_space",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            self_dependent=bundle.self_dependent_latent_states,
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
            asymmetric_coordination=bundle.gsr_asymmetric_coordination,
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
            uuid="gsr_group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=[transformation],
        )

        return group

    def _create_vocalic_group(self, bundle: BrainBundle) -> ComponentGroup:
        """
        Creates component groups for the vocalic component.

        @param bundle: config bundle holding information on how to parameterize the modules.
        @return: a group to be added to the model.
        """
        vocalic_metadata: SerialMetadata = self.metadata["speech_vocalics"]
        state_space = SerialGaussianLatentComponent(
            uuid="vocalic_state_space",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=bundle.num_vocalic_features,
            self_dependent=bundle.self_dependent_latent_states,
            mean_mean_a0=bundle.vocalic_mean_mean_a0,
            sd_mean_a0=bundle.vocalic_sd_mean_a0,
            sd_sd_a=bundle.vocalic_sd_sd_a,
            share_mean_a0_across_subjects=bundle.vocalic_share_mean_a0_across_subjects,
            share_sd_a_across_subjects=bundle.vocalic_share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=bundle.vocalic_share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=bundle.vocalic_share_sd_a_across_dimensions,
            dimension_names=bundle.vocalic_feature_names,
            sampling_time_scale_density=bundle.vocalic_sampling_time_scale_density,
            allow_sampled_subject_repetition=bundle.vocalic_allow_sampled_subject_repetition,
            fix_sampled_subject_sequence=bundle.vocalic_fix_sampled_subject_sequence,
            time_steps_in_coordination_scale=(
                vocalic_metadata.time_steps_in_coordination_scale
            ),
            prev_time_same_subject=vocalic_metadata.prev_time_same_subject,
            prev_time_diff_subject=vocalic_metadata.prev_time_diff_subject,
            subject_indices=vocalic_metadata.subject_indices,
            mean_a0=bundle.vocalic_mean_a0,
            sd_a=bundle.vocalic_sd_a,
            initial_samples=bundle.initial_vocalic_state_space_samples,
            asymmetric_coordination=bundle.asymmetric_coordination,
        )

        observation = SerialGaussianObservation(
            uuid="speech_vocalics",
            pymc_model=self.pymc_model,
            num_subjects=bundle.num_subjects,
            dimension_size=bundle.num_vocalic_features,
            sd_sd_o=bundle.vocalic_sd_sd_o,
            share_sd_o_across_subjects=bundle.vocalic_share_sd_o_across_subjects,
            share_sd_o_across_dimensions=bundle.vocalic_share_sd_o_across_dimensions,
            dimension_names=bundle.vocalic_feature_names,
            observed_values=vocalic_metadata.normalized_observations,
            time_steps_in_coordination_scale=(
                vocalic_metadata.time_steps_in_coordination_scale
            ),
            subject_indices=vocalic_metadata.subject_indices,
            sd_o=bundle.vocalic_sd_o,
        )

        group = ComponentGroup(
            uuid="vocalic_group",
            pymc_model=self.pymc_model,
            latent_component=state_space,
            observations=[observation],
            transformations=None,
        )

        return group

    def _create_semantic_group(self, bundle: BrainBundle) -> ComponentGroup:
        """
        Creates component groups for the semantic link component.

        @param bundle: config bundle holding information on how to parameterize the modules.
        @return: a group to be added to the model.
        """
        semantic_link_metadata: NonSerialMetadata = self.metadata.get(
            "semantic_link", None
        )
        if (
                semantic_link_metadata.time_steps_in_coordination_scale is not None
                and len(semantic_link_metadata.time_steps_in_coordination_scale) > 0
        ):
            # We only add the semantic link module if there's evidence.
            observed_semantic_links = SpikeObservation(
                uuid="semantic_link",
                pymc_model=self.pymc_model,
                num_subjects=bundle.num_subjects,
                sd_sd_s=bundle.semantic_link_sd_sd_s,
                dimension_name="linked",
                sampling_time_scale_density=bundle.vocalic_sampling_time_scale_density,
                time_steps_in_coordination_scale=(
                    semantic_link_metadata.time_steps_in_coordination_scale
                ),
                sd_s=bundle.semantic_link_sd_s,
                observed_values=semantic_link_metadata.observed_values,
            )

            semantic_link_group = ComponentGroup(
                uuid="semantic_link_group",
                pymc_model=self.pymc_model,
                latent_component=NullLatentComponent(),
                observations=[observed_semantic_links],
            )

            return semantic_link_group

        return None

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

            new_bundle.fnirs_mean_a0.append(
                idata.get_posterior_samples(
                    f"fnirs_state_space{suffix}_mean_a0", samples_idx
                )
            )
            new_bundle.fnirs_sd_a.append(
                idata.get_posterior_samples(
                    f"fnirs_state_space{suffix}_sd_a", samples_idx
                )
            )
            new_bundle.fnirs_sd_o.append(
                idata.get_posterior_samples(f"fnirs{suffix}_sd_o", samples_idx)
            )
            new_bundle.initial_fnirs_state_space_samples.append(
                idata.get_posterior_samples(f"fnirs_state_space{suffix}", samples_idx)
            )

        if new_bundle.common_cause:
            new_bundle.mean_cc0 = idata.get_posterior_samples(f"common_cause_mean_a0", samples_idx)
            new_bundle.sd_cc = idata.get_posterior_samples(f"common_cause_sd_a", samples_idx)
            new_bundle.initial_common_cause_samples = idata.get_posterior_samples(
                f"common_cause", samples_idx)

        new_bundle.gsr_mean_a0 = idata.get_posterior_samples(
            "gsr_state_space_mean_a0", samples_idx
        )
        new_bundle.gsr_sd_a = idata.get_posterior_samples(
            "gsr_state_space_sd_a", samples_idx
        )
        new_bundle.gsr_sd_o = idata.get_posterior_samples("gsr_sd_o", samples_idx)
        new_bundle.initial_gsr_state_space_samples = idata.get_posterior_samples(
            "gsr_state_space", samples_idx
        )

        new_bundle.vocalic_mean_a0 = idata.get_posterior_samples(
            "vocalic_state_space_mean_a0", samples_idx
        )
        new_bundle.vocalic_sd_a = idata.get_posterior_samples(
            "vocalic_state_space_sd_a", samples_idx
        )
        new_bundle.vocalic_sd_o = idata.get_posterior_samples(
            "speech_vocalics_sd_o", samples_idx
        )
        new_bundle.initial_vocalic_state_space_samples = idata.get_posterior_samples(
            "vocalic_state_space", samples_idx
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

        if config_bundle.include_semantic:
            new_bundle.semantic_link_sd_s = idata.get_posterior_samples(
                "semantic_link_sd_s", samples_idx
            )

        return new_bundle
