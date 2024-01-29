import streamlit as st

from coordination.webapp.component.inference_progress import InferenceProgress
from coordination.webapp.component.run_id_multi_selection import \
    RunIDMultiSelection
from coordination.webapp.constants import (INFERENCE_RESULTS_DIR_STATE_KEY,
                                           NUM_LAST_RUNS, REFRESH_RATE)
from coordination.webapp.utils import get_inference_run_ids


class Progress:
    """
    This class represents a page to monitor the progress of different inference runs.
    """

    def __init__(self, page_key: str):
        """
        Creates the page object.

        @param page_key: unique identifier for the page. Components of the page will append to
            this key to form their keys.
        """
        self.page_key = page_key

    def create_page(self):
        """
        Creates the page by adding different components to the screen.
        """
        col1, col2 = st.columns(2)
        with col1:
            num_runs = st.number_input(
                f"{self.page_key}_num_runs", value=NUM_LAST_RUNS, min_value=1
            )
        with col2:
            run_id_multi_selection_component = RunIDMultiSelection(
                component_key=f"{self.page_key}_run_id_multi_selection",
                all_run_ids=get_inference_run_ids(
                    st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY]
                ),
            )
            run_id_multi_selection_component.create_component()

        col1, col2, col3 = st.columns(3)
        with col2:
            display_experiment_progress = st.toggle("Display Experiment Progress", value=False)
        with col3:
            display_sub_experiment_progress = st.toggle("Display Sub-experiment Progress",
                                                       value=False)

        inference_progress_component = None
        with col1:
            if st.toggle("Monitor progress"):
                inference_progress_component = InferenceProgress(
                    component_key=f"{self.page_key}_inference_progress",
                    inference_dir=st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY],
                    refresh_rate=REFRESH_RATE,
                    display_experiment_progress=display_experiment_progress,
                    display_sub_experiment_progress=display_sub_experiment_progress,
                )
                if run_id_multi_selection_component.selected_run_ids_:
                    inference_progress_component.preferred_run_ids = (
                        run_id_multi_selection_component.selected_run_ids_
                    )
                else:
                    idx = min(num_runs, len(run_id_multi_selection_component.all_run_ids))
                    inference_progress_component.preferred_run_ids = (
                        run_id_multi_selection_component.all_run_ids[:idx]
                    )

        if inference_progress_component is not None:
            inference_progress_component.create_component()
