import uuid

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown


class ExperimentIDMultiSelectionComponent:
    """
    Represents a component that displays a collection of experiments for a given inference run
    for multi-selection.
    """

    def __init__(self, all_experiment_ids: List[str]):
        """
        Creates the component.

        @param all_experiment_ids: list of all experiment IDs to display for selection.
        """
        self.all_experiment_ids = all_experiment_ids

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_experiment_ids_ = None

    def create_component(self):
        """
        Creates area in the screen for selection of multiple experiment ids.
        """
        self.all_experiment_ids = st.multiselect(
            label="Experiment IDs",
            key=str(uuid.uuid4()),
            options=self.all_experiment_ids
        )
