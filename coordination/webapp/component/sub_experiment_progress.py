import json
import os
from typing import Dict, Optional

import streamlit as st

from coordination.inference.inference_run import InferenceRun
from coordination.webapp.widget.progress_bar import ProgressBar


class SubExperimentProgress:
    """
    Represents a component that displays the progress of an inference run for sub-experiments of
    an experiment if an experiment has sub-experiments. Otherwise, it treats the experiment as a
    sub-experiment of itself. Progress bars are created for each chain individually and overall
    progress is monitored by peeking log files. This element does not update itself periodically,
    it needs a call to create_component to update its content.
    """

    def __init__(
        self,
        inference_run: InferenceRun,
        experiment_id: str,
        sub_experiment_id: Optional[str] = None,
        display_sub_experiment_progress: bool = True,
    ):
        """
        Creates the component.

        @param inference_run: object containing info about an inference run.
        @param experiment_id: experiment id from the inference run.
        @param sub_experiment_id: optional sub-experiment ID.
        @param display_sub_experiment_progress: whether to display the progress of all the
            sub-experiments of all the experiments in the inference run.
        """
        self.inference_run = inference_run
        self.experiment_id = experiment_id
        self.sub_experiment_id = sub_experiment_id
        self.display_sub_experiment_progress = display_sub_experiment_progress

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.status_ = ""
        self.total_num_divergences_ = 0

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

        If an experiment has sub-experiments, their individual status will be displayed.
        """
        logs = self._read_logs()
        self.status_ = self._peek_logs(logs)

        divergence_progress_container = None
        if self.display_sub_experiment_progress:
            if self.sub_experiment_id:
                if self.status_ == "in_progress":
                    progress_emoji = ":hourglass:"
                elif self.status_ == "success":
                    progress_emoji = ":white_check_mark:"
                elif self.status_ == "failed":
                    progress_emoji = ":x:"
                else:
                    progress_emoji = ":question:"

                st.write(f"### :orange[{self.sub_experiment_id} {progress_emoji}]")
            divergence_progress_container = st.container()

            col1, col2 = st.columns([0.05, 0.95])
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
        total_num_samples = (
            total_samples_per_chain * self.inference_run.execution_params["num_chains"]
        )
        sorted_chain_names = sorted(list(progress_info["step"].keys()))
        self.total_num_divergences_ = 0

        chains_in_progress = []
        for chain in sorted_chain_names:
            self.total_num_divergences_ += progress_info["num_divergences"][chain]
            if progress_info["step"][chain] < total_samples_per_chain:
                chains_in_progress.append(chain)

        if self.display_sub_experiment_progress:
            if chains_in_progress:
                st.write("#### :violet[Samples]")
                for chain in chains_in_progress:
                    ProgressBar(
                        items_name=f"samples in {chain}",
                        current_value=progress_info["step"][chain],
                        maximum_value=total_samples_per_chain,
                    ).create()

                st.write("#### :violet[Divergences]")
                for chain in chains_in_progress:
                    ProgressBar(
                        items_name=f"divergences in {chain}",
                        current_value=progress_info["num_divergences"][chain],
                        maximum_value=total_samples_per_chain,
                    ).create()

            with divergence_progress_container:
                ProgressBar(
                    items_name="divergences.",
                    current_value=self.total_num_divergences_,
                    maximum_value=total_num_samples,
                ).create()

    def _read_logs(self) -> Optional[str]:
        """
        Gets the logs of the inference run for the experiment ID from a log file saved under the
        experiment ID folder. We read the log in memory since it's expected to be small.

        @return: content of the log file if such file exists.
        """
        experiment_dir = f"{self.inference_run.run_dir}/{self.experiment_id}"
        if self.sub_experiment_id:
            experiment_dir += f"/ppa/{self.sub_experiment_id}"
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
            if logs.rfind("ERROR") > logs.rfind("INFO"):
                # If there's an INFO after error is because another run is being attempted.
                return "failed"

        if logs.find("SUCCESS") >= 0 or logs.find("Duration") >= 0:
            return "success"

        return "in_progress"

    def _read_progress_info(self) -> Optional[Dict[str, float]]:
        """
        Reads progress json file for the experiment ID.

        @return: progress dictionary.
        """
        experiment_dir = f"{self.inference_run.run_dir}/{self.experiment_id}"
        if self.sub_experiment_id:
            experiment_dir += f"/ppa/{self.sub_experiment_id}"
        progress_filepath = f"{experiment_dir}/progress.json"
        if not os.path.exists(progress_filepath):
            return None

        try:
            with open(progress_filepath, "r") as f:
                progress_dict = json.load(f)
        except Exception:
            return None

        return progress_dict
