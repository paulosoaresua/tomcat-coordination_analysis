import json
import os
import subprocess
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import streamlit as st
from pkg_resources import resource_string

from coordination.common.constants import (DEFAULT_BURN_IN, DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_INFERENCE_JOBS,
                                           DEFAULT_NUM_JOBS_PER_INFERENCE,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED, DEFAULT_TARGET_ACCEPT)
from coordination.model.builder import ModelBuilder
from coordination.model.config.mapper import DataMapper
from coordination.webapp.constants import (INFERENCE_PARAMETERS_DIR,
                                           INFERENCE_TMP_DIR)
from coordination.webapp.widget.drop_down import DropDown


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
        self.inference_dir = inference_dir

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """

        # Creates a dropdown to allow users to select default parameters from a json file.
        selected_default_execution_params_file = DropDown(
            label="Default Execution Parameters",
            key=f"{self.component_key}_default_execution_parameters_dropdown",
            options=InferenceExecution._get_saved_execution_parameter_files(),
        ).create()
        execution_params = InferenceExecution._load_saved_execution_params(
            selected_default_execution_params_file
        )

        if not execution_params:
            execution_params = (
                InferenceExecution._assemble_default_execution_params_dict()
            )

        execution_params = self._create_execution_params_area(execution_params)

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            self._create_inference_triggering_area(execution_params)
        with col_right:
            self._create_execution_params_saving_area(execution_params)

    @staticmethod
    def _get_saved_execution_parameter_files() -> List[str]:
        """
        Gets the list of filenames with saved execution parameters.

        @return: list of files with saved execution parameters.
        """
        if os.path.exists(INFERENCE_PARAMETERS_DIR):
            saved_params_list = sorted(
                [
                    f
                    for f in os.listdir(INFERENCE_PARAMETERS_DIR)
                    if os.path.isfile(f"{INFERENCE_PARAMETERS_DIR}/{f}")
                ]
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
        execution_params = dict(
            seed=DEFAULT_SEED,
            burn_in=DEFAULT_BURN_IN,
            num_samples=DEFAULT_NUM_SAMPLES,
            num_chains=DEFAULT_NUM_CHAINS,
            num_jobs_per_inference=DEFAULT_NUM_JOBS_PER_INFERENCE,
            num_inference_jobs=DEFAULT_NUM_INFERENCE_JOBS,
            nuts_init_method=DEFAULT_NUTS_INIT_METHOD,
            target_accept=DEFAULT_TARGET_ACCEPT,
            model=None,
            data_filepath=None,
            model_params={},
            data_mapping={"mappings": []},
        )

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
                value=default_execution_params["seed"],
            )
            execution_params["burn_in"] = st.number_input(
                label="Burn-in",
                key=f"{self.component_key}_burn_in",
                value=default_execution_params["burn_in"],
            )
            execution_params["num_samples"] = st.number_input(
                label="Number of Samples",
                key=f"{self.component_key}_num_samples",
                value=default_execution_params["num_samples"],
            )
            execution_params["num_chains"] = st.number_input(
                label="Number of Chains",
                key=f"{self.component_key}_num_chains",
                value=default_execution_params["num_chains"],
            )
            execution_params["num_jobs_per_inference"] = st.number_input(
                label="Number of Jobs per Inference (typically = number of chains)",
                key=f"{self.component_key}_num_jobs_per_inference",
                value=default_execution_params["num_jobs_per_inference"],
            )
            execution_params["num_inference_jobs"] = st.number_input(
                label="Number of Inference Jobs (how many experiment batches in parallel)",
                key=f"{self.component_key}_num_inference_jobs",
                value=default_execution_params["num_inference_jobs"],
            )
            execution_params["nuts_init_method"] = st.text_input(
                label="NUTS Initialization Method",
                key=f"{self.component_key}_nuts_init_method",
                value=default_execution_params["nuts_init_method"],
            )
            execution_params["target_accept"] = st.number_input(
                label="Target Accept",
                key=f"{self.component_key}_target_accept",
                value=default_execution_params["target_accept"],
            )

        with tab2:
            model_options = sorted(list(ModelBuilder.MODELS))
            selected_model_index = (
                model_options.index(default_execution_params["model"])
                if default_execution_params["model"]
                else 0
            )
            execution_params["model"] = st.selectbox(
                label="Model",
                key=f"{self.component_key}_model",
                index=selected_model_index,
                options=model_options,
            )
            execution_params["data_filepath"] = st.text_input(
                label="Data Filepath",
                key=f"{self.component_key}_data_filepath",
                value=default_execution_params["data_filepath"],
            )
            execution_params["model_params"] = st.text_area(
                label="Model Parameters",
                key=f"{self.component_key}_model_params",
                value=json.dumps(default_execution_params["model_params"], indent=4),
                height=10,
            )
            execution_params["data_mapping"] = st.text_area(
                label="Data Mapping",
                key=f"{self.component_key}_data_mapping",
                value=json.dumps(default_execution_params["data_mapping"], indent=4),
                height=10,
            )
            # Show data mapping schema below the text area for context
            schema = json.loads(
                resource_string(
                    "coordination", "schema/data_mapper_schema.json"
                ).decode("utf-8")
            )
            st.write("Data Mapping Schema:")
            st.json(schema, expanded=False)

        return execution_params

    @staticmethod
    def _save_execution_params(execution_params: Dict[str, Any], filename: str):
        """
        Saves a dictionary of execution parameters for later usage.

        @param execution_params: a dictionary with values of execution parameters.
        @param filename: name of the file to save the execution parameters in.
        @raise Exception: if either the json provided as model parameters or data mapping is
            invalid.
        """
        execution_params_copy = deepcopy(execution_params)
        if filename and len(filename) > 0:
            # Transform model_params and data_mapping strings to json objects before saving.
            try:
                execution_params_copy["model_params"] = json.loads(
                    execution_params["model_params"]
                )
            except Exception:
                raise Exception(
                    "Invalid model parameters. Make sure to enter a valid json object."
                )

            try:
                execution_params_copy["data_mapping"] = json.loads(
                    execution_params["data_mapping"]
                )
                DataMapper(execution_params_copy["data_mapping"])
            except Exception:
                raise Exception(
                    "Invalid data mapping. Make sure to enter a valid json object."
                )

            os.makedirs(INFERENCE_PARAMETERS_DIR, exist_ok=True)
            with open(f"{INFERENCE_PARAMETERS_DIR}/{filename}.json", "w") as f:
                json.dump(execution_params_copy, f)
        else:
            raise Exception(
                "Please, provide a valid filename before saving the parameters."
            )

    def _create_execution_params_saving_area(self, execution_params: Dict[str, Any]):
        """
        Creates area with an input field to enter a name to save the execution parameters on the
        screen and a button to save them for later usage.

        @param execution_params: execution parameters for the inference run.
        """

        filename = st.text_input(
            label="Filename",
            key=f"{self.component_key}_execution_params_filename",
            placeholder="Enter a filename without extension",
        )
        if st.button(label="Save Parameters"):
            try:
                InferenceExecution._save_execution_params(execution_params, filename)
                with st.spinner("Saving..."):
                    # Wait a bit so there's has time for the file to be saved and loaded in the
                    # dropdown when the page refreshes.
                    time.sleep(2)
                st.success(
                    f"Execution parameters ({filename}) were saved successfully."
                )

            except Exception as ex:
                st.error(ex)

    def _create_inference_triggering_area(self, execution_params: Dict[str, Any]):
        """
        Creates area with a button to trigger an inference run with some execution parameters.

        @param execution_params: execution parameters for the inference run.
        """

        if st.button(label="Run Inference"):
            # Save the model parameters and data mapping dictionaries to a temporary folder so
            # that the inference script can read them.
            os.makedirs(f"{INFERENCE_TMP_DIR}", exist_ok=True)

            model_params_filepath = f"{INFERENCE_TMP_DIR}/params_dict.json"
            with open(model_params_filepath, "w") as f:
                json.dump(json.loads(execution_params["model_params"]), f)

            data_mapping_filepath = f"{INFERENCE_TMP_DIR}/data_mapping.json"
            with open(data_mapping_filepath, "w") as f:
                json.dump(json.loads(execution_params["data_mapping"]), f)

            command = (
                'PYTHONPATH="." '
                "./bin/run_inference "
                f'--out_dir="{self.inference_dir}" '
                f'--evidence_filepath="{execution_params["data_filepath"]}" '
                f'--model_name="{execution_params["model"]}" '
                f'--data_mapping_filepath="{data_mapping_filepath}" '
                f'--model_params_dict_filepath="{model_params_filepath}" '
                f'--seed={execution_params["seed"]} '
                f'--burn_in={execution_params["burn_in"]} '
                f'--num_samples={execution_params["num_samples"]} '
                f'--num_chains={execution_params["num_chains"]} '
                f'--num_jobs_per_inference={execution_params["num_jobs_per_inference"]} '
                f'--num_inference_jobs={execution_params["num_inference_jobs"]} '
                f'--nuts_init_method="{execution_params["nuts_init_method"]}" '
                f'--target_accept={execution_params["target_accept"]}'
            )

            with st.spinner("Wait for it..."):
                outputs = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                ).communicate()
                output = "".join([o.decode("utf-8") for o in outputs])
        else:
            output = ""

        st.text_area(label="Terminal Output", disabled=True, value=output)
