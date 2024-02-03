import streamlit as st

from coordination.webapp.constants import (WEBAPP_RUN_DIR_STATE_KEY,
                                           INFERENCE_RESULTS_DIR_STATE_KEY,
                                           EVALUATIONS_DIR_STATE_KEY,
                                           DATA_DIR_STATE_KEY)


class Header:
    """
    This class represents a header component comprised of an input field for entering an
    inference directory. It saves the inference directory to the state variable so it can be
    directly accessed by any component of the page.
    """

    @staticmethod
    def create_component():
        """
        Creates an input field for entering an inference directory.
        """
        st.write("**Directories**")
        st.write(
            f"Inference: *:blue[{st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY]}]*"
        )
        st.write(f"Evaluations: *:blue[{st.session_state[EVALUATIONS_DIR_STATE_KEY]}]*")
        st.write(f"Data: *:blue[{st.session_state[DATA_DIR_STATE_KEY]}]*")
        st.write(f"Temporary Files: *:blue[{st.session_state[WEBAPP_RUN_DIR_STATE_KEY]}]*")

        with st.expander("Configure Directories", expanded=False):
            inference_dir = st.text_input(
                label="Inference",
                value=st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY],
            )

            evaluations_dir = st.text_input(
                label="Evaluations",
                value=st.session_state[EVALUATIONS_DIR_STATE_KEY],
            )

            data_dir = st.text_input(
                label="Data",
                value=st.session_state[DATA_DIR_STATE_KEY],
            )

            tmp_dir = st.text_input(
                label="Temporary Files",
                value=st.session_state[WEBAPP_RUN_DIR_STATE_KEY],
            )

            submit = st.button(label="Update Directories")
            if submit:
                st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY] = inference_dir
                st.session_state[EVALUATIONS_DIR_STATE_KEY] = evaluations_dir
                st.session_state[DATA_DIR_STATE_KEY] = data_dir
                st.session_state[WEBAPP_RUN_DIR_STATE_KEY] = tmp_dir
                st.rerun()
