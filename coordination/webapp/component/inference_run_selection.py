import streamlit as st

from coordination.inference.inference_run import InferenceRun
from coordination.webapp.utils import get_inference_run_ids
from coordination.webapp.widget.drop_down import DropDown


class InferenceRunSelection:
    """
    Represents a component that displays a collection of inference runs to choose from and the
    associated exec params json object one an inference run is selected.
    """

    def __init__(self, component_key: str, inference_dir: str):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_dir: directory where inference runs were saved.
        """
        self.component_key = component_key
        self.inference_dir = inference_dir

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_inference_run_ = None

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """
        run_id = DropDown(
            label="Inference run ID",
            key=f"{self.component_key}_run_id_dropdown",
            options=get_inference_run_ids(self.inference_dir),
        ).create()

        if run_id:
            self.selected_inference_run_ = InferenceRun(self.inference_dir, run_id,
                                                        data_dir=st.session_state[
                                                            DATA_DIR_STATE_KEY])

            if self.selected_inference_run_.execution_params:
                # Display execution parameter under the drop down
                col1, col2 = st.columns([0.12, 0.88])
                with col1:
                    st.write("**Execution Parameters:**")
                with col2:
                    st.json(
                        self.selected_inference_run_.execution_params, expanded=False
                    )
