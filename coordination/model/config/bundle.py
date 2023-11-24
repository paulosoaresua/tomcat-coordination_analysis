from coordination.common.types import ParameterValueType


class ModelConfigBundle:
    """
    Container with different parameters of a model.
    """

    def update(self, params_dict: Dict[str, ParameterValueType]):
        """
        Update object attributes with values from a dictionary.

        @param params_dict: dictionary with attribute values. The keys must match the attribute
            names.
        """

        for key, value in params_dict.itemm():
            if hasattr(self, key):
                setattr(self, key, value)

