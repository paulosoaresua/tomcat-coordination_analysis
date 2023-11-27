import json

import asyncio
import streamlit as st
import os
from collections import OrderedDict

from coordination.webapp.constants import REFRESH_RATE, INFERENCE_PARAMETERS_DIR
from coordination.common.constants import (DEFAULT_BURN_IN,
                                           DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED,
                                           DEFAULT_TARGET_ACCEPT)


def create_run_page():
    """
    Creates the run page layout.
    """

    inference_results = st.text_input(label="Inference Results Directory",
                                      value=st.session_state["inference_results_dir"])
    submit = st.button(label="Update Directory")
    if submit:
        st.session_state["inference_results_dir"] = inference_results

    st.header("Trigger Inference")
    inference_pane = st.empty()
    _populate_inference_pane(inference_pane)

    st.header("Progress")
    progress_pane = st.empty()
    asyncio.run(_populate_progress_pane(progress_pane, refresh_rate=REFRESH_RATE))


def _populate_inference_pane(inference_pane: st.container):
    """
    Populates the inference pane where one can trigger a new inference process.

    @param inference_pane: container to place the elements of the pane into.
    """
    with inference_pane:
        with st.container():
            saved_params_list = [None]
            if os.path.exists(INFERENCE_PARAMETERS_DIR):
                saved_params_list.extend(sorted(
                    [f for f in os.listdir(INFERENCE_PARAMETERS_DIR) if
                     os.path.isfile(f"{INFERENCE_PARAMETERS_DIR}/{f}")]))
            default_parameter_file = st.selectbox(
                "Default parameters",
                options=saved_params_list,
                format_func=lambda
                    x: x if x else "-- Select a json file with default parameters for inference --")

            default_parameters = {}
            if default_parameter_file:
                with open(f"{INFERENCE_PARAMETERS_DIR}/{default_parameter_file}", "r") as f:
                    default_parameters = json.load(f)

            tab1, tab2 = st.columns(2)
            inference_execution_params = {}

            with tab1:
                inference_execution_params["seed"] = st.number_input(
                    label="Seed",
                    value=default_parameters.get("seed", DEFAULT_SEED))
                inference_execution_params["burn_in"] = st.number_input(
                    label="Burn-in",
                    value=default_parameters.get("burn_in", DEFAULT_BURN_IN))
                inference_execution_params["num_samples"] = st.number_input(
                    label="Number of samples",
                    value=default_parameters.get("num_samples", DEFAULT_NUM_SAMPLES))
                inference_execution_params["num_chains"] = st.number_input(
                    label="Number of chains",
                    value=default_parameters.get("num_chains", DEFAULT_NUM_CHAINS))
                inference_execution_params["num_inference_jobs"] = st.number_input(
                    label="Number of Inference Jobs",
                    value=default_parameters.get("num_inference_jobs", DEFAULT_NUM_JOBS))
                inference_execution_params["nuts_init_method"] = st.text_input(
                    label="NUTS init method",
                    value=default_parameters.get("nuts_init_method", DEFAULT_NUTS_INIT_METHOD))
                inference_execution_params["target_accept"] = st.number_input(
                    label="Target accept",
                    value=default_parameters.get("target_accept", DEFAULT_TARGET_ACCEPT))

            with tab2:
                model_params_dict = st.text_area(
                    label="Model parameters",
                    value=json.dumps(default_parameters.get("model_params", {}), indent=4))
                data_mapping = st.text_area(
                    label="Data mapping",
                    value=json.dumps(default_parameters.get("data_mapping", {}), indent=4))

                st.divider()
                filename = st.text_input(label="Filename",
                                         placeholder="Enter a filename without extension")
                save_parameters = st.button(label="Save parameters")

                if save_parameters:
                    if filename and len(filename) > 0:
                        try:
                            inference_execution_params["model_params"] = json.loads(
                                model_params_dict)
                        except:
                            st.error("Invalid model parameters. Make sure to enter a valid json "
                                     "object.")

                        try:
                            inference_execution_params["data_mapping"] = json.loads(data_mapping)
                        except:
                            st.error(
                                "Invalid data mapping. Make sure to enter a valid json object.")

                        with open(f"{INFERENCE_PARAMETERS_DIR}/{filename}.json", "w") as f:
                            json.dump(inference_execution_params, f)
                    else:
                        st.error("Please, provide a valid filename before saving the inference "
                                 "execution parameters.")

            st.divider()
            submit = st.button(label="Run Inference")
            st.text_area(label="Terminal Output", disabled=True)


async def _populate_progress_pane(progress_pane: st.container, refresh_rate: int):
    """
    Populates the progress pane where one can see the progress of the different inference
    processes.

    @param progress_pane: container to place the elements of the pane into.
    @param refresh_rate: number of seconds to wait before refreshing the pane.
    """
    while True:
        inference_dir = st.session_state["inference_results_dir"]
        with progress_pane:
            with st.container():
                status_text = st.empty()
                if os.path.exists(inference_dir):
                    run_ids = [run_id for run_id in sorted(os.listdir(inference_dir), reverse=True)
                               if
                               os.path.isdir(f"{inference_dir}/{run_id}")]
                    for i, run_id in enumerate(run_ids):
                        with open(f"{inference_dir}/{run_id}/execution_params.json", "r") as f:
                            execution_params = json.load(f)
                        total_samples_per_chain = execution_params["burn_in"] + execution_params[
                            "num_samples"]

                        with st.expander(run_id, expanded=(i == 0)):
                            st.json(execution_params, expanded=False)
                            for experiment_id in os.listdir(f"{inference_dir}/{run_id}"):
                                if not os.path.isdir(f"{inference_dir}/{run_id}/{experiment_id}"):
                                    continue

                                with open(
                                        f"{inference_dir}/{run_id}/{experiment_id}/progress.json",
                                        "r") as f:
                                    progress = json.load(f)

                                st.write(f"## {experiment_id}")
                                for key, value in OrderedDict(progress["samples"]).items():
                                    perc_value = value / total_samples_per_chain
                                    text = f"{key} - {value} out of {total_samples_per_chain} - " \
                                           f"{100.0 * perc_value}%"
                                    st.progress(perc_value, text=text)
                else:
                    st.write(f"The directory `{inference_dir}` does not exist.")

                for i in range(refresh_rate, 0, -1):
                    status_text.write(f"**Refreshing in :red[{i} seconds].**")
                    await asyncio.sleep(1)
