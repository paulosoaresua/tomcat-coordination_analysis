from typing import Dict, Optional

from coordination.common.types import ParameterValueType
from coordination.model.config.vocalic import VocalicConfigBundle
from coordination.model.model import Model
from coordination.model.real.vocalic import VocalicModel
from coordination.model.real.vocalic_semantic_link import \
    VocalicSemanticLinkModel
from coordination.model.synthetic.conversation import ConversationModel
from coordination.model.synthetic.spring import SpringModel


class ModelBuilder:
    """
    This class is responsible from instantiating a concrete model object from its name.
    """

    @staticmethod
    def build(
        model_name: str,
        model_params_dict: Optional[Dict[str, ParameterValueType]] = None,
    ) -> Model:
        """
        Gets an instance of the model.

        @param model_name: name of the model.
        @param model_params_dict: an optional dictionary containing argument values to be passed
            to the model's __init__ method.
        @return: an instance of the model.
        """

        if model_name == "conversation":
            return ConversationModel(**model_params_dict)

        if model_name == "spring":
            return SpringModel(**model_params_dict)

        if model_name == "vocalic":
            config_bundle = VocalicConfigBundle()
            if model_params_dict:
                config_bundle.update(model_params_dict)
            return VocalicModel(config_bundle=config_bundle)

        if model_name == "vocalic_semantic":
            return VocalicSemanticLinkModel(**model_params_dict)

        raise ValueError(f"Invalid model ({model_name}).")
