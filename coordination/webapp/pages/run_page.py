import json

import asyncio
import streamlit as st
import os
from collections import OrderedDict

from coordination.webapp.constants import REFRESH_RATE, INFERENCE_PARAMETERS_DIR, RUN_DIR
from coordination.common.constants import (DEFAULT_BURN_IN,
                                           DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED,
                                           DEFAULT_TARGET_ACCEPT)
import subprocess
from coordination.webapp.utils import (get_inference_run_ids,
                                       get_saved_execution_parameter_files,
                                       create_dropdown_with_default_selection,
                                       get_execution_params)


def create_run_page():
    """
    Creates the run page layout.
    """

    st.header("Trigger Inference")
    inference_pane = st.empty()
    _populate_inference_pane(inference_pane)

    st.header("Progress")
    if st.checkbox("Monitor progress"):
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

            default_parameter_file = create_dropdown_with_default_selection(
                label="Default execution parameters",
                key="inference_default_exec_params",
                values=get_saved_execution_parameter_files()
            )

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
                inference_execution_params["num_jobs_per_inference"] = st.number_input(
                    label="Number of Jobs per Inference (typically = number of chains)",
                    value=default_parameters.get("num_inference_jobs", DEFAULT_NUM_JOBS))
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
            if submit:
                # Save the parameter dictionaries in a tmp folder to that the inference script
                # can read them.
                os.makedirs(f"{RUN_DIR}/tmp/config", exist_ok=True)
                with open(f"{RUN_DIR}/tmp/config/data_mapping.json", "w") as f:
                    json.dump(json.loads(data_mapping), f)

                with open(f"{RUN_DIR}/tmp/config/params_dict.json", "w") as f:
                    json.dump(json.loads(model_params_dict), f)

                out_dir = st.session_state["inference_results_dir"]
                command = (
                    'PYTHONPATH="." '
                    './bin/run_inference '
                    f'--out_dir="{out_dir}" '
                    '--evidence_filepath="data/asist_data.csv" '
                    '--model_name="vocalic" '
                    f'--data_mapping_filepath="{RUN_DIR}/tmp/config/data_mapping.json" '
                    f'--model_params_dict_filepath="{RUN_DIR}/tmp/config/params_dict.json" '
                    f'--seed={inference_execution_params["seed"]} '
                    f'--burn_in={inference_execution_params["burn_in"]} '
                    f'--num_samples={inference_execution_params["num_samples"]} '
                    f'--num_chains={inference_execution_params["num_chains"]} '
                    f'--num_jobs_per_inference='
                    f'{inference_execution_params["num_jobs_per_inference"]} '
                    f'--num_inference_jobs={inference_execution_params["num_inference_jobs"]} '
                    f'--nuts_init_method={inference_execution_params["nuts_init_method"]} '
                    f'--target_accept={inference_execution_params["target_accept"]}'
                )

                with st.spinner('Wait for it...'):
                    outputs = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True
                    ).communicate()
                    output = "".join([o.decode("utf-8") for o in outputs])
            else:
                output = ""

            st.text_area(label="Terminal Output", disabled=True, value=output)


async def _populate_progress_pane(progress_pane: st.container, refresh_rate: int):
    """
    Populates the progress pane where one can see the progress of the different inference
    processes.

    It's not possible to have widgets that require unique keys in this pane because the widget
    keys are not cleared until the next run. We could keep creating different keys but this would
    cause memory leakage as the keys would be accumulated in the run context.

    @param progress_pane: container to place the elements of the pane into.
    @param refresh_rate: number of seconds to wait before refreshing the pane.
    """
    while True:
        inference_dir = st.session_state["inference_results_dir"]
        with progress_pane:
            with st.container():
                # The status contains a countdown showing how many seconds until the next refresh.
                # It is properly filled in the end of this function.
                status_text = st.empty()

                run_ids = get_inference_run_ids()
                if len(run_ids) > 0:
                    for i, run_id in enumerate(run_ids):
                        execution_params_dict = get_execution_params(run_id)
                        if not execution_params_dict:
                            continue

                        total_samples_per_chain = execution_params_dict["burn_in"] + \
                                                  execution_params_dict["num_samples"]
                        with st.expander(run_id, expanded=(i == 0)):
                            run_info_container = st.container()

                            # Display progress of each experiment
                            num_finished_experiments = 0
                            num_experiments_with_error = 0
                            experiment_ids = sorted(execution_params_dict["experiment_ids"])
                            for experiment_id in experiment_ids:
                                experiment_dir = f"{inference_dir}/{run_id}/{experiment_id}"

                                # From the logs, see if the execution failed, finished
                                # successfully, or it's still going on so we can put a mark beside
                                # the experiment id on the screen.
                                experiment_progress_emoji = ":hourglass:"
                                log_filepath = f"{experiment_dir}/log.txt"
                                logs = ""
                                if os.path.exists(log_filepath):
                                    with open(log_filepath, "r") as f:
                                        # We read the log in memory since it's expected to be
                                        # small.
                                        logs = f.read()
                                        if logs.find("ERROR") >= 0:
                                            experiment_progress_emoji = ":x:"
                                            num_experiments_with_error += 1
                                        elif logs.find("SUCCESS") >= 0:
                                            experiment_progress_emoji = ":white_check_mark:"
                                            num_finished_experiments += 1

                                st.write(f"## {experiment_id} {experiment_progress_emoji}")
                                st.json({"logs": logs}, expanded=False)

                                progress_filepath = f"{experiment_dir}/progress.json"
                                if not os.path.exists(progress_filepath):
                                    continue

                                with open(progress_filepath, "r") as f:
                                    progress_dict = json.load(f)
                                for key, value in OrderedDict(progress_dict["step"]).items():
                                    # Display progress bar for each chain.
                                    perc_value = value / total_samples_per_chain
                                    text = (f"{key} - {value} out of {total_samples_per_chain} - "
                                            f"{100.0 * perc_value}%")
                                    st.progress(perc_value, text=text)

                            with run_info_container:
                                perc_completion = num_finished_experiments / len(experiment_ids)
                                if perc_completion < 1:
                                    outputs = subprocess.Popen(
                                        "tmux ls",
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        shell=True
                                    ).communicate()
                                    open_tmux_sessions = "".join(
                                        [o.decode("utf-8") for o in outputs])
                                    if open_tmux_sessions.find(
                                            execution_params_dict["tmux_session_name"]) < 0:
                                        st.write(
                                            "**:red[No tmux session for the run found. The inference "
                                            "process was killed]**.")

                                if num_experiments_with_error > 0:
                                    st.write(f":x: {num_experiments_with_error} experiments "
                                             f"finished with an error.")

                                # Percentage of completion
                                text = (f"{num_finished_experiments} out of {len(experiment_ids)} "
                                        f"experiments - {100.0 * perc_completion}%")
                                st.progress(perc_completion, text=text)

                                # Display collapsed json with the execution params
                                st.json(execution_params_dict, expanded=False)

                # Wait a few seconds and Update countdown
                for i in range(refresh_rate, 0, -1):
                    status_text.write(f"**Refreshing in :red[{i} seconds].**")
                    await asyncio.sleep(1)
