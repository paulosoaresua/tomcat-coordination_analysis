from coordination.model.config_bundle.bundle import ModelConfigBundle
from coordination.model.config_bundle.vocalic import (VocalicSemanticLinkConfigBundle,
                                                      VocalicConfigBundle)
from coordination.model.config_bundle.vocalic_2d import (
    Vocalic2DSemanticLinkConfigBundle, Vocalic2DConfigBundle)
from coordination.model.real.vocalic import VocalicModel
from coordination.model.real.vocalic_2d import Vocalic2DModel
from coordination.model.real.vocalic_2d_semantic_link import \
    Vocalic2DSemanticLinkModel
from coordination.model.real.vocalic_semantic_link import \
    VocalicSemanticLinkModel
from coordination.model.template import ModelTemplate
from coordination.model.config_bundle.fnirs import FNIRSConfigBundle
from coordination.model.real.fnirs import FNIRSModel

MODELS = {
    "vocalic",
    "vocalic_semantic",
    "fnirs"
}


class ModelBuilder:
    """
    This class is responsible from instantiating a concrete model object from its name.
    """

    @staticmethod
    def build_bundle(model_name: str) -> ModelConfigBundle:
        """
        Gets an instance of a model config bundle.

        @param model_name: name of the model.
        @raise ValueError: if the model name is not in the list of valid models.
        @return: an instance of the model config bundle.
        """
        if model_name not in MODELS:
            raise ValueError(f"Invalid model ({model_name}).")

        if model_name == "vocalic":
            return VocalicConfigBundle()

        if model_name == "vocalic_semantic":
            return VocalicSemanticLinkConfigBundle()

        if model_name == "fnirs":
            return FNIRSConfigBundle()

    @staticmethod
    def build_model(model_name: str, config_bundle: ModelConfigBundle) -> ModelTemplate:
        """
        Gets an instance of the model.

        @param model_name: name of the model.
        @param config_bundle: a config bundle containing model's parameter values.
        @raise ValueError: if the model name is not in the list of valid models.
        @return: an instance of the model.
        """
        if model_name not in MODELS:
            raise ValueError(f"Invalid model ({model_name}).")

        if model_name == "vocalic":
            return VocalicModel(config_bundle=config_bundle)

        if model_name == "vocalic_semantic":
            return VocalicSemanticLinkModel(config_bundle=config_bundle)

        if model_name == "fnirs":
            return FNIRSModel(config_bundle=config_bundle)
