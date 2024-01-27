from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional

import pandas as pd

from coordination.inference.inference_data import InferenceData
from coordination.inference.model_variable import ModelVariableInfo
from coordination.model.template import ModelTemplate
from coordination.model.builder import ModelBuilder
from coordination.model.config_bundle.bundle import ModelConfigBundle
from coordination.model.config_bundle.mapper import DataMapper
from coordination.common.config import settings


class InferenceRun:
    """
    Represents a container for information related to an inference run.
    """

    def __init__(self, inference_dir: str, run_id: str):
        """
        Creates an inference run object.

        @param inference_dir: directory where the inference run if saved.
        @param run_id: ID of the inference run.
        """
        self.inference_dir = inference_dir
        self.run_id = run_id

        self.execution_params = None
        # Load execution parameters for the run
        execution_params_filepath = f"{inference_dir}/{run_id}/execution_params.json"
        if os.path.exists(execution_params_filepath):
            with open(execution_params_filepath, "r") as f:
                self.execution_params = json.load(f)

    @property
    def ppa(self) -> bool:
        """
        Gets whether the run is a PPA. If affirmative, multiple inferences were performed for each
        experiment with a different number of time steps being fit.

        @return: True if PPA was performed.
        """
        if self.execution_params is None:
            return False

        return self.execution_params.get("do_ppa", False)

    @property
    def model(self) -> ModelTemplate:
        """
        Gets an instance of the model used to fit the data in an inference run.

        @return: model.
        """
        if self.execution_params is None:
            return None

        bundle = type(self.config_bundle)(**self.execution_params["model_config_bundle"])
        return ModelBuilder.build_model(self.execution_params["model_name"], bundle)

    @property
    def data_mapper(self) -> DataMapper:
        """
        Gets an instance of the mapper used to map data into config bundle parameters in the
        inference run.

        @return: data mapper.
        """
        if self.execution_params is None:
            return None

        return DataMapper(self.execution_params["data_mapper"])

    @property
    def config_bundle(self) -> ModelConfigBundle:
        """
        Gets an instance of the config bundle used in the inference run.

        @return: config bundle.
        """
        if self.execution_params is None:
            return None

        return ModelBuilder.build_bundle(self.execution_params["model_name"])

    @property
    def data(self) -> pd.DataFrame:
        """
        Gets an instance of the data frame containing the data for all experiments in the
        inference run.

        @return: data.
        """
        if self.execution_params is None:
            return None

        lb = self.execution_params["evidence_filepath"].rfind("/") + 1
        data_filename = self.execution_params["evidence_filepath"][lb:]

        data_filepath = f"{settings.data_dir}/{data_filename}"
        return pd.read_csv(data_filepath)

    @property
    def run_dir(self) -> str:
        """
        Gets the directory of the inference run.

        @return: directory of the inference run.
        """
        return f"{self.inference_dir}/{self.run_id}"

    @property
    def experiment_ids(self) -> List[str]:
        """
        Gets a list of experiment IDs evaluated in the inference run.

        @return: list of experiment IDs
        """

        if "experiment_ids" not in self.execution_params:
            # Older inference runs don't have this field. We use all the directory names under the
            # run as experiment ids.
            return [d for d in os.listdir(self.run_dir) if os.path.isdir(f"{self.run_dir}/{d}")]

        return self.execution_params["experiment_ids"] if self.execution_params else []

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
        experiment_dir = f"{self.run_dir}/{experiment_id}"
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
            "posterior_predictive": [],
        }
        for mode in [
            "prior_predictive",
            "posterior",
            "posterior_predictive",
            "observed_data",
        ]:
            if mode not in idata.trace:
                continue

            for var_name in idata.trace[mode].data_vars:
                dim_coordinate = f"{var_name}_dimension"
                dim_names = (
                    idata.trace[mode][dim_coordinate].data.tolist()
                    if dim_coordinate in idata.trace[mode]
                    else []
                )

                var_info = ModelVariableInfo(
                    variable_name=var_name,
                    inference_mode=mode,
                    dimension_names=dim_names,
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

        return variables_dict

    def has_active_tmux_session(self) -> bool:
        """
        Checks whether there's an active TMUX session for the run.

        @return: True if theres an active TMUX session for the run.
        """
        outputs = subprocess.Popen(
            "tmux ls",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        ).communicate()
        open_tmux_sessions = "".join([o.decode("utf-8") for o in outputs])

        return open_tmux_sessions.find(self.execution_params["tmux_session_name"]) >= 0
