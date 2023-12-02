import streamlit as st

from coordination.webapp.component.inference_run_selection import \
    InferenceRunSelection
from coordination.webapp.component.experiment_id_multi_selection import \
    ExperimentIDMultiSelection
from coordination.webapp.component.model_variable_selection import \
    ModelVariableSelection
from coordination.webapp.component.inference_results import InferenceResults
from coordination.webapp.entity.inference_run import InferenceRun
from coordination.webapp.constants import INFERENCE_RESULTS_DIR_STATE_KEY


class NewRun:
    """
    This class represents a page to start a new inference run in a tmux session in the machine
    where the app is running.
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
        st.header("Trigger Inference")
        inference_pane = st.empty()
        _populate_inference_pane(inference_pane)

        st.header("Progress")
        if st.checkbox("Monitor progress"):
            progress_pane = st.empty()
            asyncio.run(_populate_progress_pane(progress_pane, refresh_rate=REFRESH_RATE))
