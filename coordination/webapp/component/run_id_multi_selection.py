from typing import List

import streamlit as st


class RunIDMultiSelection:
    """
    Represents a component that displays a collection of inference runs for multi-selection.
    """

    def __init__(self, component_key: str, all_run_ids: List[str]):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param all_run_ids: list of all run IDs to display for selection.
        """
        self.component_key = component_key
        self.all_run_ids = all_run_ids

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_run_ids_ = None

    def create_component(self):
        """
        Creates a multi-selector in the screen for selection of multiple run ids.
        """
        self.selected_run_ids_ = st.multiselect(
            label="Run IDs",
            key=f"{self.component_key}_run_ids_multiselect",
            options=self.all_run_ids,
        )
