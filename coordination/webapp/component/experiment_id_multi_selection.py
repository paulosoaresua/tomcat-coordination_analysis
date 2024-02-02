from typing import List

import streamlit as st


class ExperimentIDMultiSelection:
    """
    Represents a component that displays a collection of experiments for a given inference run
    for multi-selection.
    """

    def __init__(self, component_key: str, all_experiment_ids: List[str]):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param all_experiment_ids: list of all experiment IDs to display for selection.
        """
        self.component_key = component_key
        self.all_experiment_ids = all_experiment_ids

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_experiment_ids_ = None

    def create_component(self):
        """
        Creates a multi-selector in the screen for selection of multiple experiment ids.
        """
        self.selected_experiment_ids_ = st.multiselect(
            label="Experiment IDs",
            key=f"{self.component_key}_experiment_ids_multiselect",
            options=self.all_experiment_ids,
        )
