import uuid
from typing import List, Optional

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown
from coordination.webapp.entity.inference_run import InferenceRun
import os
from coordination.webapp.constants import INFERENCE_PARAMETERS_DIR
from coordination.common.constants import (DEFAULT_BURN_IN, DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS_PER_INFERENCE,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED, DEFAULT_TARGET_ACCEPT,
                                           DEFAULT_NUM_INFERENCE_JOBS)
from copy import deepcopy
from coordination.model.config.mapper import DataMapper


class InferenceExecution:
    """
    Represents a component that displays a collection of parameters to be passed for execution of
    a new inference run.
    """

    def __init__(self, component_key: str, inference_dir: str):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_dir: directory where inference runs were saved.
        """
        self.component_key = component_key

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """

        # Creates a dropdown to allow users to select default parameters from a json file.
        selected_default_execution_params_file = DropDown(
            label="Default Execution Parameters",
            key=f"{self.component_key}_default_execution_parameters_dropdown",
            options=InferenceExecution._get_saved_execution_parameter_files()
        ).create()
        execution_params = InferenceExecution._load_saved_execution_params(
            selected_default_execution_params_file)

        if not execution_params:
            execution_params = InferenceExecution._assemble_default_execution_params_dict()

        execution_params = self._create_execution_params_area(execution_params)

        st.divider()

        filename = st.text_input(
            label="Filename",
            key=f"{self.component_key}_execution_params_filename",
            placeholder="Enter a filename without extension"
        )
        save_execution_params_button = st.button(label="Save Parameters")
        if save_execution_params_button:
            InferenceExecution._save_execution_params(execution_params, filename)

        st.divider()

    @staticmethod
    def _get_saved_execution_parameter_files() -> List[str]:
        """
        Gets the list of filenames with saved execution parameters.

        @return: list of files with saved execution parameters.
        """
        if os.path.exists(INFERENCE_PARAMETERS_DIR):
            saved_params_list = sorted(
                [f for f in os.listdir(INFERENCE_PARAMETERS_DIR) if
                 os.path.isfile(f"{INFERENCE_PARAMETERS_DIR}/{f}")]
            )
            return saved_params_list

        return []

    @staticmethod
    def _load_saved_execution_params(filename: str) -> Optional[Dict[str, Any]]:
        """
        Gets a dictionary with saved execution params.

        @param filename: name of the file.
        @return: dictionary with saved execution params.
        """
        if filename:
            with open(f"{INFERENCE_PARAMETERS_DIR}/{filename}", "r") as f:
                return json.load(f)

        return None

    @staticmethod
    def _assemble_default_execution_params_dict() -> Optional[Dict[str, Any]]:
        """
        Assembles a dictionary of execution params with default values.

        @return: a dictionary of execution params.
        """
        execution_params = dict(seed=DEFAULT_SEED,
                                burn_in=DEFAULT_BURN_IN,
                                num_samples=DEFAULT_NUM_SAMPLES,
                                num_chains=DEFAULT_NUM_CHAINS,
                                num_jobs_per_inference=DEFAULT_NUM_JOBS_PER_INFERENCE,
                                num_inference_jobs=DEFAULT_NUM_INFERENCE_JOBS,
                                nuts_init_method=DEFAULT_NUTS_INIT_METHOD,
                                target_accept=DEFAULT_TARGET_ACCEPT,
                                model_params={},
                                data_mapping={"mappings": []})

        return execution_params

    def _create_execution_params_area(self, default_execution_params: Dict[str, Any]):
        """
        Creates area on the screen with fields for entering the execution parameters of the
        inference run.

        @param execution_params: a dictionary with default values of execution parameters.
        @return: a dictionary of updated values of execution parameters.
        """
        execution_params = {}

        tab1, tab2 = st.columns(2)
        with tab1:
            execution_params["seed"] = st.number_input(
                label="Seed",
                key=f"{self.component_key}_seed",
                value=default_execution_params["seed"]
            )
            execution_params["burn_in"] = st.number_input(
                label="Burn-in",
                key=f"{self.component_key}_burn_in",
                value=default_execution_params["burn_in"]
            )
            execution_params["num_samples"] = st.number_input(
                label="Number of Samples",
                key=f"{self.component_key}_num_samples",
                value=default_execution_params["num_samples"]
            )
            execution_params["num_chains"] = st.number_input(
                label="Number of Chains",
                key=f"{self.component_key}_num_chains",
                value=default_execution_params["num_chains"]
            )
            execution_params["num_jobs_per_inference"] = st.number_input(
                label="Number of Jobs per Inference (typically = number of chains)",
                key=f"{self.component_key}_num_jobs_per_inference",
                value=default_execution_params["num_jobs_per_inference"]
            )
            execution_params["num_inference_jobs"] = st.number_input(
                label="Number of Inference Jobs (how many experiment batches in parallel)",
                key=f"{self.component_key}_num_inference_jobs",
                value=default_execution_params["num_inference_jobs"]
            )
            execution_params["nuts_init_method"] = st.number_input(
                label="NUTS Initialization Method",
                key=f"{self.component_key}_nuts_init_method",
                value=default_execution_params["nuts_init_method"]
            )
            execution_params["target_accept"] = st.number_input(
                label="Target Accept",
                key=f"{self.component_key}_target_accept",
                value=default_execution_params["target_accept"]
            )

        with tab2:
            execution_params["model_params"] = st.number_input(
                label="Model Parameters",
                key=f"{self.component_key}_model_params",
                value=default_execution_params["model_params"]
            )
            execution_params["data_mapping"] = st.number_input(
                label="Data Mapping",
                key=f"{self.component_key}_data_mapping",
                value=default_execution_params["data_mapping"]
            )

        return execution_params

    @staticmethod
    def _save_execution_params(execution_params: Dict[str, Any], filename: str):
        """
        Saves a dictionary of execution parameters for later usage.

        @param execution_params: a dictionary with values of execution parameters.
        @param filename: name of the file to save the execution parameters in.
        """
        execution_params_copy = deepcopy(execution_params)
        if filename and len(filename) > 0:
            # Transform model_params and data_mapping strings to json objects before saving.
            try:
                execution_params_copy["model_params"] = json.loads(
                    execution_params["model_params"]
                )
            except Exception as ex:
                st.error(
                    f"Invalid model parameters. Make sure to enter a valid json object. {ex}"
                )

            try:
                execution_params_copy["data_mapping"] = json.loads(
                    execution_params["data_mapping"]
                )
                DataMapper(execution_params_copy["data_mapping"])
            except Exception as ex:
                st.error(
                    f"Invalid data mapping. Make sure to enter a valid json object. {ex}"
                )

            with open(f"{INFERENCE_PARAMETERS_DIR}/{filename}.json", "w") as f:
                json.dump(execution_params_copy, f)
        else:
            st.error(
                "Please, provide a valid filename before saving the parameters."
            )
