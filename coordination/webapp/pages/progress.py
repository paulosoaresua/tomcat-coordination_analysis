import streamlit as st

from coordination.webapp.component.inference_progress import InferenceProgress
from coordination.webapp.constants import (INFERENCE_RESULTS_DIR_STATE_KEY,
                                           REFRESH_RATE)


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
        inference_progress_component = InferenceProgress(
            component_key=f"{self.page_key}_inference_progress",
            inference_dir=st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY],
            refresh_rate=REFRESH_RATE,
        )

        if st.button("Refresh"):
            st.rerun()

        inference_progress_component.create_component()
