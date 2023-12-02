import uuid
from typing import List

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown
from coordination.webapp.entity.inference_run import InferenceRun
import os


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

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_inference_run_ = InferenceRun(inference_dir=inference_dir, run_id=None)

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """
        self.selected_inference_run_.run_id = DropDown(
            label="Inference run ID",
            key=f"{self.component_key}_run_id_dropdown",
            options=self._get_inference_run_ids()).create()

    def _get_inference_run_ids(self) -> List[str]:
        """
        Gets a list of inference run IDs from the list of directories under an inference folder.

        @return: list of inference run ids.
        """
        if os.path.exists(self.selected_inference_run_.inference_dir):
            run_ids = [run_id for run_id in os.listdir(self.selected_inference_run_.inference_dir)
                       if os.path.isdir(f"{self.selected_inference_run_.inference_dir}/{run_id}")]

            # Display on the screen from the most recent to the oldest.
            return sorted(run_ids, reverse=True)

        return []
