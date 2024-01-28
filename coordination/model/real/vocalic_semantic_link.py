from typing import Optional

import pymc as pm

from coordination.model.config_bundle.vocalic import \
    VocalicSemanticLinkConfigBundle
from coordination.model.model import Model
from coordination.model.template import ModelTemplate
from coordination.module.component_group import ComponentGroup
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.null_latent_component import \
    NullLatentComponent
from coordination.module.latent_component.serial_gaussian_latent_component import \
    SerialGaussianLatentComponent
from coordination.module.observation.serial_gaussian_observation import \
    SerialGaussianObservation
from coordination.module.observation.spike_observation import SpikeObservation
from coordination.module.transformation.mlp import MLP
from coordination.model.real.vocalic import VocalicModel
from coordination.metadata.non_serial import NonSerialMetadata
from copy import deepcopy
import numpy as np


class VocalicSemanticLinkModel(VocalicModel):
    """
    This class represents a model where subjects are talking to each other and their speech
    vocalics are observed as they finish talking as well as semantic links between subsequent
    utterances from different subjects.
    """

    def __init__(
            self,
            config_bundle: VocalicSemanticLinkConfigBundle,
            pymc_model: Optional[pm.Model] = None,
    ):
        """
        Creates a vocalic + semantic link model.

        @param config_bundle: container for the different parameters of the vocalic + semantic
            link model.
        @param pymc_model: a PyMC model instance where modules are to be created at. If not
            provided, it will be created along with this model instance.
        """

        super().__init__(pymc_model=pymc_model, config_bundle=config_bundle)

    def _register_metadata(self, config_bundle: VocalicSemanticLinkConfigBundle):
        """
        Add entries to the metadata dictionary from values filled in a config bundle. This will
        allow adjustment of time steps later if we want to fit/sample less time steps than the
        informed in the original config bundle.
        """
        super()._register_metadata(config_bundle)

        if "semantic_link" in self.metadata:
            metadata: NonSerialMetadata = self.metadata["semantic_link"]
            metadata.time_steps_in_coordination_scale = (
                config_bundle.semantic_link_time_steps_in_coordination_scale)
        else:
            self.metadata["semantic_link"] = NonSerialMetadata(
                time_steps_in_coordination_scale=(
                    config_bundle.semantic_link_time_steps_in_coordination_scale),
                observed_values=None,
                normalization_method=None
            )

    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """
        super()._create_model_from_config_bundle()
        self._model.uuid = "vocalic_semantic_link_model"

        semantic_link_metadata = self.metadata.get("semantic_link", None)
        if semantic_link_metadata and semantic_link_metadata.time_steps_in_coordination_scale is \
                not None and len(semantic_link_metadata.time_steps_in_coordination_scale) > 0:
            # We only add the semantic link module if there's evidence.

            bundle = self._get_adjusted_bundle()

            observed_semantic_links = SpikeObservation(
                uuid="semantic_link",
                pymc_model=self.pymc_model,
                num_subjects=bundle.num_subjects,
                a_p=bundle.a_p,
                b_p=bundle.b_p,
                dimension_name="linked",
                sampling_time_scale_density=bundle.sampling_time_scale_density,
                time_steps_in_coordination_scale=(
                    semantic_link_metadata.time_steps_in_coordination_scale
                ),
                p=bundle.p,
            )

            semantic_link_group = ComponentGroup(
                uuid="semantic_link_group",
                pymc_model=self.pymc_model,
                latent_component=NullLatentComponent(),
                observations=[observed_semantic_links],
            )
            self._model.component_groups.append(semantic_link_group)

    def _get_adjusted_bundle(self) -> VocalicSemanticLinkConfigBundle:
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

                # Map each one of the semantic lint time step to the new scale. We can do this
                # because the time steps with semantic link is a subset of the time steps with
                # vocalics.
                time_mapping = {t: new_t for new_t, t in
                                enumerate(self.config_bundle.time_steps_in_coordination_scale)}
                for i, t in enumerate(
                        self.config_bundle.semantic_link_time_steps_in_coordination_scale):
                    bundle.semantic_link_time_steps_in_coordination_scale[i] = time_mapping[t]

        return self.new_config_bundle_from_time_step_info(bundle)

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
        new_bundle = super().new_config_bundle_from_posterior_samples(config_bundle, idata,
                                                                      num_samples, seed)

        new_bundle.p = idata.get_posterior_samples("semantic_link_p", samples_idx)

        return new_bundle
