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
from coordination.webapp.component.experiment_progress import ExperimentProgress
from coordination.webapp.widget.progress_bar import ProgressBar


class InferenceRunProgress:
    """
    Represents a component that displays the progress of an inference run. It displays the
    progress of each experiment under the run and a global status for the run itself.
    """

    def __init__(self, inference_run: InferenceRun):
        """
        Creates the component.

        @param inference_run: object containing info about an inference run.
        """
        self.inference_run = inference_run

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.status_ = False

    def create_component(self):
        """
        Show the global progress of the run and individual progress of each experiment in the run.
        """
        # Stores global info about a run...
        run_info_container = st.container()

        # Display progress of each experiment
        num_experiments_succeeded = 0
        num_experiments_in_progress = 0
        num_experiments_not_started = 0
        num_experiments_failed = 0
        experiment_ids = sorted(self.inference_run.experiment_ids)
        for experiment_id in experiment_ids:
            experiment_progress_component = ExperimentProgress(inference_run=self.inference_run,
                                                               experiment_id=experiment_id)
            experiment_progress_component.create_component()
            num_experiments_succeeded += experiment_progress_component.succeeded
            num_experiments_in_progress += experiment_progress_component.in_progress
            num_experiments_not_started += experiment_progress_component.unknown
            num_experiments_failed += experiment_progress_component.failed

        with run_info_container:
            # Display collapsed json with the execution params
            col1, col2 = st.columns([0.12, 0.88])
            with col1:
                st.write("**Execution Parameters:**")
            with col2:
                st.json(self.inference_run.execution_params, expanded=False)

            # Check for interruptions
            if num_experiments_succeeded < len(experiment_ids):
                if not self.inference_run.has_active_tmux_session():
                    st.write(
                        "*:red[No tmux session for the run found. The "
                        "inference process was killed]*."
                    )

            # Display progress bars

            # Success
            left_col_width_perc = 0.02
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":white_check_mark:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_succeeded,
                    maximum_value=len(experiment_ids)
                ).create()

            # Fail
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":x:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_failed,
                    maximum_value=len(experiment_ids)
                ).create()

            # In Progress
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":hourglass:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_in_progress,
                    maximum_value=self.inference_run.execution_params["num_inference_jobs"]
                ).create()

            # Not Started
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":question:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_not_started,
                    maximum_value=len(experiment_ids)
                ).create()
