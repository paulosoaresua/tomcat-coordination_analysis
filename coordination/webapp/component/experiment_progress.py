import json
import os
from typing import Dict, Optional

import streamlit as st

from coordination.inference.inference_run import InferenceRun
from coordination.webapp.widget.progress_bar import ProgressBar
from coordination.webapp.component.sub_experiment_progress import SubExperimentProgress
import numpy as np


class ExperimentProgress:
    """
    Represents a component that displays the progress of an inference run for an experiment.
    Progress bars are created for each chain individually and overall progress is monitored by
    peeking log files. This element does not update itself periodically, it needs a call to
    create_component to update its content.
    """

    def __init__(self, inference_run: InferenceRun, experiment_id: str,
                 display_experiment_progress: bool = True,
                 display_sub_experiment_progress: bool = True):
        """
        Creates the component.

        @param inference_run: object containing info about an inference run.
        @param experiment_id: experiment id from the inference run.
        @param display_experiment_progress: whether to display the progress of all the experiments
            in the inference run.
        @param display_sub_experiment_progress: whether to display the progress of all the
            sub-experiments of all the experiments in the inference run.
        """
        self.inference_run = inference_run
        self.experiment_id = experiment_id
        self.display_experiment_progress = display_experiment_progress
        self.display_sub_experiment_progress = display_sub_experiment_progress

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.status_ = False
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
        experiment_title_container = None
        if self.display_experiment_progress:
            experiment_title_container = st.container()

        all_status = set()
        all_divergences = []
        if self.inference_run.ppa:
            for sub_exp_id in self.inference_run.get_sub_experiment_ids(self.experiment_id):
                sub_experiment_progress = SubExperimentProgress(
                    self.inference_run,
                    self.experiment_id,
                    sub_exp_id,
                    self.display_sub_experiment_progress)
                sub_experiment_progress.create_component()
                all_status.add(sub_experiment_progress.status_)
                all_divergences.append(sub_experiment_progress.total_num_divergences_)

            if len(all_divergences) > 0:
                self.total_num_divergences_ = int(np.mean(all_divergences))
        else:
            sub_experiment_progress = SubExperimentProgress(
                self.inference_run,
                self.experiment_id,
                display_sub_experiment_progress=self.display_experiment_progress)
            sub_experiment_progress.create_component()
            all_status.add(sub_experiment_progress.status_)
            self.total_num_divergences_ = sub_experiment_progress.total_num_divergences_

        # Update status based on individual status of the sub experiments.
        if "failed" in all_status:
            self.status_ = "failed"
        elif "in_progress" in all_status:
            self.status_ = "in_progress"
        elif {"success"} == all_status:
            self.status_ = "success"
        else:
            self.status_ = "no_logs"

        if self.display_experiment_progress:
            if self.status_ == "in_progress":
                progress_emoji = ":hourglass:"
            elif self.status_ == "success":
                progress_emoji = ":white_check_mark:"
            elif self.status_ == "failed":
                progress_emoji = ":x:"
            else:
                progress_emoji = ":question:"

            with experiment_title_container:
                st.write(f"## {self.experiment_id} {progress_emoji}")
