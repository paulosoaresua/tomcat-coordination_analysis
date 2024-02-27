from copy import deepcopy

from coordination.model.config_bundle.brain import BrainBundle
from coordination.model.config_bundle.bundle import ModelConfigBundle
from coordination.model.config_bundle.vocalic import VocalicConfigBundle
from coordination.model.real.brain import BrainModel
from coordination.model.real.vocalic import VocalicModel
from coordination.model.template import ModelTemplate

MODELS = {
    "vocalic",
    "vocalic_semantic",
    "vocalic_2d",
    "vocalic_2d_semantic",
    "brain",
    "brain_gsr",
    "brain_vocalic",
    "brain_gsr_vocalic",
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

        if "brain" in model_name:
            bundle = BrainBundle()
            bundle.include_gsr = False
            bundle.include_vocalic = False
            if "gsr" in model_name:
                bundle.include_gsr = True

            if "vocalic" in model_name:
                bundle.include_vocalic = True

            return bundle
        else:
            if "vocalic" in model_name:
                bundle = VocalicConfigBundle()
                bundle.state_space_2d = False
                bundle.include_semantic = False
                if "vocalic_2d" in model_name:
                    bundle.state_space_2d = True
                if "semantic" in model_name:
                    bundle.include_semantic = True
                return bundle

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

        if "brain" in model_name:
            bundle = deepcopy(config_bundle)
            bundle.include_gsr = False
            bundle.include_vocalic = False
            if "gsr" in model_name:
                bundle.include_gsr = True
            if "vocalic" in model_name:
                bundle.include_vocalic = True

            return BrainModel(bundle)
        else:
            if "vocalic" in model_name:
                bundle = deepcopy(config_bundle)
                bundle.state_space_2d = False
                bundle.include_semantic = False
                if "vocalic_2d" in model_name:
                    bundle.state_space_2d = True
                if "semantic" in model_name:
                    bundle.include_semantic = True

                return VocalicModel(bundle)
