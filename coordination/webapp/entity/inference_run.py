from typing import Any, List, Optional, Dict
from coordination.webapp.entity.model_variable import ModelVariableInfo
from coordination.inference.inference_data import InferenceData
import os
import json


class InferenceRun:
    """
    Represents a container for information related to an inference run.
    """

    def __init__(self,
                 inference_dir: str,
                 run_id: str):
        """
        Creates an inference run object.

        @param inference_dir: directory where the inference run if saved.
        @param run_id: ID of the inference run.
        """
        self.inference_dir = inference_dir
        self.run_id = run_id

    @property
    def execution_params_dict(self) -> Optional[Dict[str, Any]]:
        """
        Gets a dictionary of execution params for an inference run if it exists.

        @return: dictionary of execution params.
        """
        execution_params_filepath = f"{self.inference_dir}/{self.run_id}/execution_params.json"
        if os.path.exists(execution_params_filepath):
            with open(execution_params_filepath, "r") as f:
                return json.load(f)

        return None

    @property
    def experiment_ids(self) -> List[str]:
        """
        Gets a list of experiment IDs evaluated in the inference run.

        @return: list of experiment IDs
        """

        return self.execution_params_dict["experiment_ids"] if self.execution_params_dict else []

    @property
    def sample_inference_data(self) -> Optional[InferenceData]:
        """
        Gets one inference data object among any of the experiments in the inference run or None
        if one cannot be found.

        @return: inference data.
        """
        for experiment_id in self.experiment_ids:
            idata = self.get_inference_data(experiment_id)
            if idata:
                return idata

        return None

    def get_inference_data(self, experiment_id: str) -> Optional[InferenceData]:
        """
        Gets inference data of an experiment.

        @param experiment_id: IF of the experiment.
        @return: inference data
        """
        experiment_dir = f"{self.inference_dir}/{self.run_id}/{experiment_id}"
        return InferenceData.from_trace_file_in_directory(experiment_dir)

    @property
    def model_variables(self) -> Dict[str, ModelVariableInfo]:
        """
        Gets a dictionary of model variables where the key is one of the following groups:
        - latent: latent data variables in the model.
        - latent_parameters: latent parameter variables in the model.
        - observed: observed data variables in the model.
        - prior_predictive: variables sampled during prior predictive check.
        - posterior_predictive: variables sampled during prior posterior check.

        @return: list of model variables and associated info.
        """
        idata = self.sample_inference_data
        if not idata:
            return {}

        variables_dict = {
            "latent": [],
            "latent_parameter": [],
            "observed": [],
            "prior_predictive": [],
            "posterior_predictive": []
        }
        for mode in ["prior_predictive", "posterior", "posterior_predictive", "observed_data"]:
            if mode not in idata.trace:
                continue

            for var_name in idata.trace[mode].data_vars:
                dim_coordinate = f"{var_name}_dimension"
                dim_names = idata.trace[mode][dim_coordinate].data.tolist() if \
                    dim_coordinate in idata.trace[mode] else []

                var_info = ModelVariableInfo(
                    variable_name=var_name,
                    inference_mode=mode,
                    dimension_names=dim_names
                )

                if mode == "posterior":
                    if idata.is_parameter(mode, var_name):
                        variables_dict["latent_parameter"].append(var_info)
                    else:
                        variables_dict["latent"].append(var_info)
                elif mode == "observed_data":
                    variables_dict["observed"].append(var_info)
                else:
                    variables_dict[mode].append(var_info)

        # for var_name in idata.trace.observed_data.data_vars:
        #     # Observed parameters can be retrieved from the inference run execution params.
        #     # We only include observed data variables here.
        #     dim_coordinate = f"{var_name}_dimension"
        #     dim_names = idata.trace.observed_data[dim_coordinate].data.tolist() if \
        #         dim_coordinate in idata.trace.observed_data else []
        #
        #     if not idata.is_parameter("observed_data", var_name):
        #         variables_dict["observed"].append(
        #             ModelVariableInfo(
        #                 variable_name=var_name,
        #                 inference_mode="observed_data",
        #                 dimension_names=dim_names
        #             )
        #         )

        return variables_dict
