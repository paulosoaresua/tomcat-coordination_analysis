import json
import os
import subprocess
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from pkg_resources import resource_string

from coordination.common.constants import (DEFAULT_BURN_IN, DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_INFERENCE_JOBS,
                                           DEFAULT_NUM_JOBS_PER_INFERENCE,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUM_TIME_POINTS_FOR_PPA,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_PPA_WINDOW, DEFAULT_SEED,
                                           DEFAULT_TARGET_ACCEPT)
from coordination.model.builder import MODELS
from coordination.model.config_bundle.mapper import DataMapper
from coordination.webapp.constants import (AVAILABLE_EXPERIMENTS_STATE_KEY,
                                           WEBAPP_RUN_DIR_STATE_KEY,
                                           DATA_DIR_STATE_KEY)
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

    @property
    def execution_params_dir(self) -> str:
        """
        Gets directory where execution parameter files are saved. Execution parameter files will
        save the entries for the fields in the new run page for later usage so we don't have to
        type them all again.

        @return: Directory where execution parameters are saved.
        """
        return f"{st.session_state[WEBAPP_RUN_DIR_STATE_KEY]}/inference/execution_params"

    @property
    def temporary_dir(self) -> str:
        """
        Gets directory where temporary model params dict (config bundle) and data mappings are
        saved for later reference by the inference script.

        @return: Directory where model params dict and data mapping of the latest run are saved.
        """
        return f"{st.session_state[WEBAPP_RUN_DIR_STATE_KEY]}/inference/tmp"

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """

        # Creates a dropdown to allow users to select default parameters from a json file.
        selected_default_execution_params_file = DropDown(
            label="Default Execution Parameters",
            key=f"{self.component_key}_default_execution_parameters_dropdown",
            options=self._get_saved_execution_parameter_files(),
        ).create()
        execution_params = self._load_saved_execution_params(
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

    def _get_saved_execution_parameter_files(self) -> List[str]:
        """
        Gets the list of filenames with saved execution parameters.

        @return: list of files with saved execution parameters.
        """
        if os.path.exists(self.execution_params_dir):
            saved_params_list = sorted(
                [
                    f
                    for f in os.listdir(self.execution_params_dir)
                    if os.path.isfile(f"{self.execution_params_dir}/{f}")
                ]
            )
            return saved_params_list

        return []

    def _load_saved_execution_params(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Gets a dictionary with saved execution params.

        @param filename: name of the file.
        @return: dictionary with saved execution params.
        """
        if filename:
            with open(f"{self.execution_params_dir}/{filename}", "r") as f:
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
            do_ppa=False,
            num_time_points_ppa=DEFAULT_NUM_TIME_POINTS_FOR_PPA,
            ppa_window=DEFAULT_PPA_WINDOW,
            model=None,
            data_filepath=None,
            experiment_ids=[],
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
            execution_params["do_ppa"] = st.toggle(
                label="PPA",
                key=f"{self.component_key}_do_ppa_checkbox",
                value=default_execution_params["do_ppa"],
            )
            if execution_params["do_ppa"]:
                col_ppa1, col_ppa2 = st.columns(2)
                with col_ppa1:
                    execution_params["num_time_points_ppa"] = st.number_input(
                        label="Number of Points for PPA",
                        key=f"{self.component_key}_num_time_points_ppa",
                        value=default_execution_params.get(
                            "num_time_points_ppa", DEFAULT_NUM_TIME_POINTS_FOR_PPA
                        ),
                    )
                with col_ppa2:
                    execution_params["ppa_window"] = st.number_input(
                        label="Window Size for PPA",
                        key=f"{self.component_key}_ppa_window",
                        value=default_execution_params.get(
                            "ppa_window", DEFAULT_PPA_WINDOW
                        ),
                    )

        with tab2:
            model_options = sorted(list(MODELS))
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
                value=default_execution_params["data_filepath"]
            )
            InferenceExecution._load_available_experiment_list(
                execution_params["data_filepath"]
            )

            if default_execution_params["experiment_ids"]:
                selected_exp_ids = [e for e in default_execution_params["experiment_ids"] if
                                    e in set(st.session_state[AVAILABLE_EXPERIMENTS_STATE_KEY])]
            else:
                selected_exp_ids = []

            execution_params["experiment_ids"] = st.multiselect(
                label="Experiments",
                key=f"{self.component_key}_experiments",
                default=selected_exp_ids,
                options=st.session_state[AVAILABLE_EXPERIMENTS_STATE_KEY],
            )
            if len(execution_params["experiment_ids"]) == 0:
                st.write("*:blue[All experiments in the dataset selected.]*")
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

            os.makedirs(self.execution_params_dir, exist_ok=True)
            with open(f"{self.execution_params_dir}/{filename}.json", "w") as f:
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
                self._save_execution_params(execution_params, filename)
                with st.spinner("Saving..."):
                    # Wait a bit so there's has time for the file to be saved and loaded in the
                    # dropdown when the page refreshes.
                    time.sleep(1)
                st.success(
                    f"Execution parameters ({filename}) were saved successfully."
                )
                time.sleep(1)
                st.rerun()

            except Exception as ex:
                st.error(ex)

    def _create_inference_triggering_area(self, execution_params: Dict[str, Any]):
        """
        Creates area with a button to trigger an inference run with some execution parameters.

        @param execution_params: execution parameters for the inference run.
        """

        if st.button(
                label="Run Inference",
                disabled=len(st.session_state[AVAILABLE_EXPERIMENTS_STATE_KEY]) == 0,
        ):
            # Save the model parameters and data mapping dictionaries to a temporary folder so
            # that the inference script can read them.
            tmp_dir = f"{st.session_state[WEBAPP_RUN_DIR_STATE_KEY]}/inference/tmp"
            os.makedirs(f"{tmp_dir}", exist_ok=True)

            model_params_filepath = f"{tmp_dir}/params_dict.json"
            with open(model_params_filepath, "w") as f:
                json.dump(json.loads(execution_params["model_params"]), f)

            data_mapping_filepath = f"{tmp_dir}/data_mapping.json"
            with open(data_mapping_filepath, "w") as f:
                json.dump(json.loads(execution_params["data_mapping"]), f)

            if len(execution_params["experiment_ids"]) > 0:
                experiment_ids_str = ",".join(execution_params["experiment_ids"])
                experiment_ids_arg = f'--experiment_ids="{experiment_ids_str}" '
            else:
                # It will default to None in the ./bin/run_inference script which means inference
                # will execute over the full list of experiments in the dataset.
                experiment_ids_arg = ""

            command = (
                'PYTHONPATH="." '
                "./bin/run_inference "
                f'--out_dir="{self.inference_dir}" '
                f'--evidence_filepath="{execution_params["data_filepath"]}" '
                f"{experiment_ids_arg}"
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
                f'--target_accept={execution_params["target_accept"]} '
                f'--do_ppa={1 if execution_params["do_ppa"] else 0}'
            )
            if "num_time_points_ppa" in execution_params:
                command += (
                    f' --num_time_points_ppa={execution_params["num_time_points_ppa"]}'
                )
            if "num_time_points_ppa" in execution_params:
                command += f' --ppa_window={execution_params["ppa_window"]}'

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

    @staticmethod
    def _load_available_experiment_list(data_filepath: str):
        """
        Loads to the session a list of available experiments in a dataset.

        @param data_filepath: path of the dataset file.
        """
        if not data_filepath:
            return

        full_data_filepath = f"{st.session_state[DATA_DIR_STATE_KEY]}/{data_filepath}"
        if os.path.exists(full_data_filepath):
            try:
                df = pd.read_csv(full_data_filepath)
                st.session_state[AVAILABLE_EXPERIMENTS_STATE_KEY] = sorted(
                    list(df["experiment_id"])
                )
            except Exception:
                st.session_state[AVAILABLE_EXPERIMENTS_STATE_KEY] = []
