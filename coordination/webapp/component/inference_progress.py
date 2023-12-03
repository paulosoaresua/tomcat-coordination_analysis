import time
import uuid
from typing import Any, Dict, List, Optional
import subprocess

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown
from coordination.webapp.entity.inference_run import InferenceRun
import os
from coordination.webapp.constants import INFERENCE_PARAMETERS_DIR, INFERENCE_TMP_DIR, \
    INFERENCE_RESULTS_DIR_STATE_KEY
from coordination.common.constants import (DEFAULT_BURN_IN, DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS_PER_INFERENCE,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED, DEFAULT_TARGET_ACCEPT,
                                           DEFAULT_NUM_INFERENCE_JOBS)
from copy import deepcopy
from coordination.model.config.mapper import DataMapper
from pkg_resources import resource_string
from coordination.model.builder import ModelBuilder
import json
import asyncio
from coordination.webapp.utils import get_inference_run_ids
from collections import OrderedDict


class InferenceProgress:
    """
    Represents a component that displays a collection of inference runs and the progress of each
    one of them.
    """

    def __init__(self, component_key: str, inference_dir: str, refresh_rate: int):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_dir: directory where inference runs were saved.
        @param refresh_rate: how many seconds to wait before updating the progress.
        """
        self.component_key = component_key
        self.inference_dir = inference_dir
        self.refresh_rate = refresh_rate

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """
        asyncio.run(self._create_progress_area())

    async def _create_progress_area(self):
        """
        Populates the progress pane where one can see the progress of the different inference
        processes.

        It's not possible to have widgets that require unique keys in this pane because the widget
        keys are not cleared until the next run. We could keep creating different keys but this would
        cause memory leakage as the keys would be accumulated in the run context.

        @param progress_pane: container to place the elements of the pane into.
        @param refresh_rate: number of seconds to wait before refreshing the pane.
        """
        progress_area = st.empty()
        while True:
            with progress_area:
                with st.container():
                    # The status contains a countdown showing how many seconds until the next refresh.
                    # It is properly filled in the end of this function.
                    status_text = st.empty()

                    run_ids = get_inference_run_ids(self.inference_dir)
                    if len(run_ids) <= 0:
                        await self._wait(status_text)
                        continue

                    for i, run_id in enumerate(run_ids):
                        inference_run = InferenceRun(
                            inference_dir=self.inference_dir,
                            run_id=run_id)

                        execution_params = inference_run.execution_params_dict
                        if not execution_params:
                            continue

                        total_samples_per_chain = (
                                execution_params["burn_in"]
                                + execution_params["num_samples"]
                        )
                        with st.expander(run_id, expanded=(i == 0)):
                            # Stores global info about a run...
                            # TODO: move this to its own container
                            run_info_container = st.container()

                            # Display progress of each experiment
                            num_finished_experiments = 0
                            num_experiments_with_error = 0
                            experiment_ids = sorted(inference_run.experiment_ids)
                            for experiment_id in experiment_ids:
                                experiment_dir = (
                                    f"{self.inference_dir}/{run_id}/{experiment_id}"
                                )

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
                                            experiment_progress_emoji = (
                                                ":white_check_mark:"
                                            )
                                            num_finished_experiments += 1

                                st.write(
                                    f"## {experiment_id} {experiment_progress_emoji}"
                                )
                                st.json({"logs": logs}, expanded=False)

                                progress_filepath = f"{experiment_dir}/progress.json"
                                if not os.path.exists(progress_filepath):
                                    continue

                                with open(progress_filepath, "r") as f:
                                    progress_dict = json.load(f)
                                for key, value in OrderedDict(
                                        progress_dict["step"]
                                ).items():
                                    # Display progress bar for each chain.
                                    perc_value = value / total_samples_per_chain
                                    text = (
                                        f"{key} - {value} out of {total_samples_per_chain} - "
                                        f"{100.0 * perc_value}%"
                                    )
                                    st.progress(perc_value, text=text)

                            with run_info_container:
                                perc_completion = num_finished_experiments / len(
                                    experiment_ids
                                )
                                if perc_completion < 1:
                                    # See if there's any tmux session for the inference run.
                                    outputs = subprocess.Popen(
                                        "tmux ls",
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        shell=True,
                                    ).communicate()
                                    open_tmux_sessions = "".join(
                                        [o.decode("utf-8") for o in outputs]
                                    )
                                    if (open_tmux_sessions.find(
                                            inference_run.execution_params_dict[
                                                "tmux_session_name"]) < 0):
                                        st.write(
                                            "**:red[No tmux session for the run found. The "
                                            "inference process was killed]**."
                                        )

                                if num_experiments_with_error > 0:
                                    st.write(
                                        f":x: {num_experiments_with_error} experiments "
                                        f"finished with an error."
                                    )

                                # Percentage of completion
                                text = (
                                    f"{num_finished_experiments} out of {len(experiment_ids)} "
                                    f"experiments - {100.0 * perc_completion}%"
                                )
                                st.progress(perc_completion, text=text)

                                # Display collapsed json with the execution params
                                st.json(inference_run.execution_params_dict, expanded=False)

            await self._wait(status_text)

    async def _wait(self, countdown_area: st.container):
        """
        Wait a few seconds and update countdown.
        """
        for i in range(self.refresh_rate, 0, -1):
            countdown_area.write(f"**Refreshing in :red[{i} seconds].**")
            await asyncio.sleep(1)
