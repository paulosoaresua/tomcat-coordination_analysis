import uuid

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown
from coordination.webapp.utils import get_inference_run_ids
from coordination.webapp.component.inference_run_selection_component import InferenceRunSelectionComponent


class InferenceRunResultsComponent:
    """
    Represents a component responsible for displaying inference results for an experiment ID
    from a specific inference run.
    """

    def __init__(self, run_id: str, experiment_ids: List[str]):
        """
        Creates a component to display inference results on the screen.

        @param run_id: ID of the inference run.
        @param experiment_ids: IDs of the experiments to plot.
        """
        self.run_id = run_id
        self.experiment_ids = experiment_ids

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_run_id_ = None
        self.execution_params_dict_ = None
        self.selected_experiment_ids_ = None

    def create_component(self):
        """
        Creates the component by adding appropriate graphical elements to the screen.
        """
        inference_run_component = InferenceRunSelectionComponent(
            inference_dir=st.session_state["inference_results_dir"])

        if not inference_run_component.selected_run_id_:
            return

        self.selected_experiment_ids_ = st.multiselect(
            label="Experiment IDs",
            key=str(uuid.uuid4()),
            options=self.execution_params_dict_["experiment_ids"]
        )

    def _create_run_id_area(self) -> bool:
        """
        Creates area in the screen for selection of an inference run id.

        @return: True if there are execution params for the inference run.
        """
        self.selected_run_id_ = DropDown(
            label="Inference run ID",
            options=get_inference_run_ids()
        )

        if self.selected_run_id_:
            # Display the execution params for the inference run
            self.execution_params_dict_ = get_execution_params(self.selected_run_id_)
            if self.execution_params_dict_:
                st.json(self.execution_params_dict_, expanded=False)

        return self.execution_params_dict_ is not None
