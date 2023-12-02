from typing import List, Optional


class ModelVariableInfo:
    """
    Represents a container to store information pertaining to a model variable.
    """

    def __init__(self,
                 name: str,
                 inference_mode: str,
                 dimension_names: Optional[List[str]]):
        """
        Creates a model variable info object.

        @param name: name of the variable in the model.
        @param inference_mode: one of posterior, prior_predictive, posterior_predictive.
        @param dimension_names: names of the dimensions of the variable if more than one.
        """
        self.name = name
        self.inference_mode = inference_mode
        self.dimension_names = dimension_names
