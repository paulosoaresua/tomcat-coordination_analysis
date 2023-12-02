import uuid

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown


class InferenceRunSelectionComponent:
    """
    Represents a component that displays a collection of inference runs to choose from and the
    associated exec params json object one an inference run is selected.
    """

    def __init__(self, inference_dir: str):
        """
        Creates the component.

        @param inference_dir: directory where inference runs are saved.
        """
        self.inference_dir = inference_dir

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_run_id_ = None
        self.execution_params_dict_ = None

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """
        self.selected_run_id_ = DropDown(
            label="Inference run ID",
            options=self._get_inference_run_ids()
        ).create()

        if self.selected_run_id_:
            # Display the execution params for the inference run
            self.execution_params_dict_ = get_execution_params(self.selected_run_id_)
            if self.execution_params_dict_:
                st.json(self.execution_params_dict_, expanded=False)

    def _get_inference_run_ids(self) -> List[str]:
        """
        Gets a list of inference run IDs from the list of directories under an inference folder.

        @return: list of inference run ids.
        """
        if os.path.exists(self.inference_dir):
            run_ids = [run_id for run_id in os.listdir(inference_dir) if
                       os.path.isdir(f"{self.inference_dir}/{run_id}")]

            # Display on the screen from the most recent to the oldest.
            return sorted(run_ids, reverse=True)

        return []
