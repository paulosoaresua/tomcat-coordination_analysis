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

        super().__init__(pymc_model, config_bundle)

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
                self.metadata["semantic_link"].time_steps_in_coordination_scale
            ),
            p=bundle.p,
        )

        semantic_link_group = ComponentGroup(
            uuid="semantic_link_group",
            pymc_model=self.pymc_model,
            latent_component=NullLatentComponent(),
            observations=[observed_semantic_links],
        )

        self._model.uuid = "vocalic_semantic_link_model"
        self._model.component_groups.append(semantic_link_group)
