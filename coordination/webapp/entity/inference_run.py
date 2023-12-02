from typing import List, Optional, Dict
from coordination.webapp.entity.model_variable import ModelVariableInfo
from coordination.inference.inference_data import InferenceData

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
        self.execution_params_dict = None

        execution_params_filepath = f"{inference_dir}/{run_id}/execution_params.json"
        if os.path.exists(execution_params_filepath):
            with open(execution_params_filepath, "r") as f:
                self.execution_params_dict = json.load(f)

    @property
    def experiment_ids(self) -> List[str]:
        """
        Gets a list of experiment IDs evaluated in the inference run.

        @return: list of experiment IDs
        """
        return self.execution_params_dict["experiment_ids"] if self.execution_params_dict else None

    @property
    def sample_inference_data(self) -> Optional[InferenceData]:
        """
        Gets one inference data object among any of the experiments in the inference run or None
        if one cannot be found.

        @return: inference data.
        """
        for experiment_id in self.experiment_ids:
            experiment_dir = f"{inference_dir}/{run_id}/{experiment_id}"
            idata = InferenceData.from_trace_file_in_directory(experiment_dir)
            if idata:
                return idata

        return None

    @property
    def model_variables(self) -> Optional[Dict[str, ModelVariableInfo]]:
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
            return None

        variables_dict = {
            "latent": [],
            "latent_parameter": [],
            "observed": [],
            "prior_predictive": [],
            "posterior_predictive": []
        }
        for mode in ["prior_predictive", "posterior", "posterior_predictive"]:
            if mode not in idata.trace:
                continue

            for var_name in idata.trace["mode"].data_vars:
                if self.is_parameter(mode, var_name):
                    variables_dict["latent_parameter"].append(
                        ModelVariableInfo(
                            name=var_name,
                            inference_mode=mode,
                            # A parameter variable can be multidimensional but they don't have named
                            # dimensions.
                            dimension_names=None
                        )
                    )
                else:
                    # This is how the dimension coordinate is defined in Module.dimension_axis_name
                    dim_coordinate = f"{variable_name}_dimension"
                    variables_dict["latent"].append(
                        ModelVariableInfo(
                            name=var_name,
                            inference_mode=mode,
                            # A parameter variable can be multidimensional but they don't have named
                            # dimensions.
                            dimension_names=idata.trace[mode][dim_coordinate].data.tolist()
                        )
                    )

        for var_name in idata.trace.observed_data.data_vars:
            # Observed parameters can be retrieved from the inference run execution params.
            # We only include observed data variables here.
            dim_coordinate = f"{variable_name}_dimension"
            if not self.is_parameter("observed_data", var_name):
                variables_dict["observed"].append(
                    ModelVariableInfo(
                        name=var_name,
                        inference_mode="observed_data",
                        dimension_names=idata.trace.posterior[dim_coordinate].data.tolist()
                    )
                )

        return variables_dict
