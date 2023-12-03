import streamlit as st

from coordination.webapp.component.inference_execution import \
    InferenceExecution
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
        inference_execution_component = InferenceExecution(
            component_key=f"{self.page_key}_inference_execution",
            inference_dir=st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY],
        )
        inference_execution_component.create_component()
