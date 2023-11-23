from typing import Dict, Optional, Union

import numpy as np

from coordination.model.model import Model
from coordination.model.synthetic.conversation import ConversationModel
from coordination.model.synthetic.spring import SpringModel
from coordination.model.real.vocalic import VocalicModel
from coordination.model.real.vocalic_semantic_link import VocalicSemanticLinkModel


class ModelBuilder:
    """
    This class is responsible from instantiating a concrete model object from its name.
    """

    @staticmethod
    def build(model_name: str,
              model_params_dict: Optional[
                  Dict[str, Union[str, int, float, np.ndarray]]] = None) -> Model:
        """
        Gets an instance of the model.

        @param model_name: name of the model.
        @param model_params_dict: an optional dictionary containing argument values to be passed
            to the model's __init__ method.
        @return: an instance of the model.
        """

        if model_name == "conversation":
            return ConversationModel.__att(**model_params_dict)

        if model_name == "spring":
            return SpringModel.__att(**model_params_dict)

        if model_name == "vocalic":
            return VocalicModel.__att(**model_params_dict)

        if model_name == "vocalic_semantic":
            return VocalicSemanticLinkModel.__att(**model_params_dict)

        raise ValueError(f"Invalid model ({model_name}).")
