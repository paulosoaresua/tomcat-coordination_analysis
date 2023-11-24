import json
from typing import Dict, List

from jsonschema import validate
import pandas as pd

from coordination.model.config.bundle import ModelConfigBundle
from coordination.common.types import ParameterValueType


class DataMapper:
    """
    This class represents a data mapper that maps values from a pandas series to a model config
    bundle.
    """

    def __init__(self,
                 data_mapping: Dict[str, List[Dist[str, Union[ParameterValueType, List[str]]]]]):
        """
        Creates a data mapper.

        @param data_mapping: list of mappings between bundle parameter name and column names in a
            pandas Series containing evidence for the model.
        @param Exception: if data_mapping is not valid.

        """
        with open("../../schema/data_mapper_schema.json") as f:
            validate(instance=data_mapping, schema=json.load(f))

        self.data_mapping = data_mapping

    def update_config_bundle(self, config_bundle: ModelConfigBundle, data: pd.Series):
        """
        Updates config bundle with values from a pandas series as instructed by the informed
        mappings.

        @param config_bundle: config bundle to be updated.
        @param data: pandas Series with the data.
        """
        for mapping in self.data_mapping["mappings"]:
            if hasattr(config_bundle, mapping["bundle_param_name"]):
                values = []
                for col in mapping["data_column_names"]:
                    values.append(data[col])

                if mapping["data_type"] == "array":
                    value = np.array(values)
                    if value.dims == 3:
                        # Subject indices come first and dimensions come in second.
                        value = value.swapaxis(0, 1)
                else:
                    value = values[0]
                setattr(config_bundle, mapping["bundle_param_name"], value)

    def validate(self, config_bundle: ModelConfigBundle, data_columns: List[str]):
        """
        Checks if config_bundle parameter names exist as well as data columns.

        @param config_bundle: config bundle to be updated.
        @param data_columns: column names in the pandas data frame.
        """
        for mapping in self.data_mapping["mappings"]:
            if not hasattr(config_bundle, mapping["bundle_param_name"]):
                raise ValueError(f"Parameter {mapping['bundle_param_name']} does not exist in the "
                                 f"model's config bundle.")

            for col in mapping["data_column_names"]:
                if col not in data_columns:
                    raise ValueError(
                        f"Column {col} does not exist in the data.")
