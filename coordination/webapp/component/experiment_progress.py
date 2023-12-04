import json
import os
from collections import OrderedDict
from typing import Dict, Optional

import streamlit as st

from coordination.webapp.entity.inference_run import InferenceRun
from coordination.webapp.widget.progress_bar import ProgressBar


class ExperimentProgress:
    """
    Represents a component that displays the progress of an inference run for an experiment.
    Progress bars are created for each chain individually and overall progress is monitored by
    peeking log files. This element does not update itself periodically, it needs a call to
    create_component to update its content.
    """

    def __init__(self, inference_run: InferenceRun, experiment_id: str):
        """
        Creates the component.

        @param inference_run: object containing info about an inference run.
        @param experiment_id: experiment id from the inference run.
        """
        self.inference_run = inference_run
        self.experiment_id = experiment_id

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.status_ = False

    @property
    def succeeded(self) -> bool:
        """
        Checks whether the inference finished successfully for the experiment.

        @return: True if it did.
        """
        return self.status_ == "success"

    @property
    def failed(self) -> bool:
        """
        Checks whether the inference failed for the experiment.

        @return: True if it did.
        """
        return self.status_ == "failed"

    @property
    def in_progress(self) -> bool:
        """
        Checks whether the inference is in progress for the experiment.

        @return: True if it is.
        """
        return self.status_ == "in_progress"

    @property
    def unknown(self) -> bool:
        """
        Checks whether the inference status is unknown for the experiment.

        @return: True if it is.
        """
        return self.status_ == "no_logs"

    def create_component(self):
        """
        Show a title with the experiment ID and an emoji indicating the overall progress. Below,
        individual progress bars for each chain is displayed.
        """
        logs = self._read_logs()
        self.status_ = self._peek_logs(logs)

        if self.status_ == "in_progress":
            progress_emoji = ":hourglass:"
        elif self.status_ == "success":
            progress_emoji = ":white_check_mark:"
        elif self.status_ == "failed":
            progress_emoji = ":x:"
        else:
            progress_emoji = ":question:"

        st.write(f"## {self.experiment_id} {progress_emoji}")
        col1, col2 = st.columns([0.03, 0.97])
        with col1:
            st.write("**Logs:**")
        with col2:
            if logs:
                st.json({"logs": logs}, expanded=False)
            else:
                st.write("*:red[No logs found.]*")

        progress_info = self._read_progress_info()
        if not progress_info:
            return

        # Use an OrderedDict such that the chains show up in order of their numbers. For instance,
        # chain1, chain 2, chain 3...
        total_samples_per_chain = (
            self.inference_run.execution_params["burn_in"]
            + self.inference_run.execution_params["num_samples"]
        )
        sorted_chain_names = sorted(list(progress_info["step"].keys()))
        for chain in sorted_chain_names:
            ProgressBar(
                items_name=f"samples in {chain}",
                current_value=progress_info["step"][chain],
                maximum_value=total_samples_per_chain,
            ).create()

    def _read_logs(self) -> Optional[str]:
        """
        Gets the logs of the inference run for the experiment ID from a log file saved under the
        experiment ID folder. We read the log in memory since it's expected to be small.

        @return: content of the log file if such file exists.
        """
        experiment_dir = f"{self.inference_run.run_dir}/{self.experiment_id}"
        log_filepath = f"{experiment_dir}/log.txt"
        logs = None
        if os.path.exists(log_filepath):
            with open(log_filepath, "r") as f:
                logs = f.read()

        return logs

    @staticmethod
    def _peek_logs(logs: Optional[str]) -> str:
        """
        From the logs, check if an inference run for the experiment ID is still in progress,
        finished successfully or failed.

        @param logs: logs of an inference run for the experiment ID.
        @return: inference run status. One of success, failed, in_progress, no_logs.
        """
        if not logs:
            return "no_logs"

        if logs.find("ERROR") >= 0:
            return "failed"

        if logs.find("SUCCESS") >= 0:
            return "success"

        return "in_progress"

    def _read_progress_info(self) -> Optional[Dict[str, float]]:
        """
        Reads progress json file for the experiment ID.

        @return: progress dictionary.
        """
        experiment_dir = f"{self.inference_run.run_dir}/{self.experiment_id}"
        progress_filepath = f"{experiment_dir}/progress.json"
        if not os.path.exists(progress_filepath):
            return None

        with open(progress_filepath, "r") as f:
            progress_dict = json.load(f)

        return progress_dict
