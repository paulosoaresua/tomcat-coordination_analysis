import json
from typing import Dict, List, Union

from ast import literal_eval
import numpy as np
import pandas as pd
from jsonschema import validate

from coordination.common.types import ParameterValueType
from coordination.model.config.bundle import ModelConfigBundle


class DataMapper:
    """
    This class represents a data mapper that maps values from a pandas series to a model config
    bundle.
    """

    def __init__(
        self,
        data_mapping: Dict[str, List[Dict[str, Union[ParameterValueType, List[str]]]]],
    ):
        """
        Creates a data mapper.

        @param data_mapping: list of mappings between bundle parameter name and column names in a
            pandas Series containing evidence for the model.
        @param Exception: if data_mapping is not valid.

        """
        with open("../coordination/schema/data_mapper_schema.json") as f:
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
                if mapping["data_type"] == "array":
                    values = []
                    for col in mapping["data_column_names"]:
                        values.append(literal_eval(data[col]))

                    value = np.array(values)
                    if value.ndim == 3:
                        # Subject indices come first and dimensions come in second.
                        value = value.swapaxis(0, 1)
                    elif len(mapping["data_column_names"]) == 1:
                        # Drop the first dimension and keep only the time series.
                        value = values[0]
                else:
                    value = data[mapping["data_column_names"][0]]

                setattr(config_bundle, mapping["bundle_param_name"], value)

    def validate(self, config_bundle: ModelConfigBundle, data_columns: List[str]):
        """
        Checks if config_bundle parameter names exist as well as data columns.

        @param config_bundle: config bundle to be updated.
        @param data_columns: column names in the pandas data frame.
        """
        for mapping in self.data_mapping["mappings"]:
            if not hasattr(config_bundle, mapping["bundle_param_name"]):
                raise ValueError(
                    f"Parameter {mapping['bundle_param_name']} does not exist in the "
                    f"model's config bundle."
                )

            for col in mapping["data_column_names"]:
                if col not in data_columns:
                    raise ValueError(f"Column {col} does not exist in the data.")
