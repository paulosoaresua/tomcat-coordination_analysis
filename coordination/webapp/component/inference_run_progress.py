import streamlit as st

from coordination.inference.inference_run import InferenceRun
from coordination.webapp.component.experiment_progress import \
    ExperimentProgress
from coordination.webapp.widget.progress_bar import ProgressBar


class InferenceRunProgress:
    """
    Represents a component that displays the progress of an inference run. It displays the
    progress of each experiment under the run and a global status for the run itself.
    """

    def __init__(
        self,
        inference_run: InferenceRun,
        display_experiment_progress: bool = True,
        display_sub_experiment_progress: bool = True,
        hide_completed_experiment: bool = True,
    ):
        """
        Creates the component.

        @param inference_run: object containing info about an inference run.
        @param display_experiment_progress: whether to display the progress of all the experiments
            in the inference run.
        @param display_sub_experiment_progress: whether to display the progress of all the
            sub-experiments of all the experiments in the inference run.
        @param hide_completed_experiment: whether to hide successfully completed experiments from
            the list.
        """
        self.inference_run = inference_run
        self.display_experiment_progress = display_experiment_progress
        self.display_sub_experiment_progress = display_sub_experiment_progress
        self.hide_completed_experiment = hide_completed_experiment

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
            experiment_progress_component = ExperimentProgress(
                inference_run=self.inference_run,
                experiment_id=experiment_id,
                display_experiment_progress=self.display_experiment_progress,
                display_sub_experiment_progress=self.display_sub_experiment_progress,
                hide_completed_experiment=self.hide_completed_experiment,
            )
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
                    maximum_value=len(experiment_ids),
                ).create()

            # Fail
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":x:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_failed,
                    maximum_value=len(experiment_ids),
                ).create()

            # In Progress
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":hourglass:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_in_progress,
                    maximum_value=self.inference_run.execution_params[
                        "num_inference_jobs"
                    ],
                ).create()

            # Not Started
            col1, col2 = st.columns([left_col_width_perc, 1 - left_col_width_perc])
            with col1:
                st.write(":question:")
            with col2:
                ProgressBar(
                    items_name="experiments",
                    current_value=num_experiments_not_started,
                    maximum_value=len(experiment_ids),
                ).create()
