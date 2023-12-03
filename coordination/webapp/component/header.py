import streamlit as st

from coordination.webapp.constants import INFERENCE_RESULTS_DIR_STATE_KEY


class Header:
    """
    This class represents a header component comprised of an input field for entering an
    inference directory. It saves the inference directory to the state variable so it can be
    directly accessed by any component of the page.
    """

    def create_component(self):
        """
        Creates an input field for entering an inference directory.
        """
        inference_results = st.text_input(
            label="Inference Results Directory",
            value=st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY],
        )
        if inference_results:
            st.write(f"In use: *:blue[{inference_results}]*")

        submit = st.button(label="Update Directory")
        if submit:
            st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY] = inference_results
