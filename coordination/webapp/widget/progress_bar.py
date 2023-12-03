from __future__ import annotations

import streamlit as st


class ProgressBar:
    """
    Creates a progress bar with numerical information about the progress.
    """

    def __init__(self, items_name: str, current_value: int, maximum_value: int):
        """
        Creates a progress bar widget.

        @param items_name: what are being counted (e.g., experiments)
        @param current_value: current progress value.
        @param maximum_value: maximum progress value.
        """
        self.items_name = items_name
        self.current_value = current_value
        self.maximum_value = maximum_value

    def create(self):
        """
        Creates the widget.
        """
        perc_completion = self.current_value / self.maximum_value
        text = (
            f"{self.current_value} out of {self.maximum_value} {self.items_name} - "
            f"{100.0 * perc_completion:.2f}%"
        )
        st.progress(perc_completion, text=text)
