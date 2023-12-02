import streamlit as st


class ExperimentResultsComponent:
    """
    Represents a component responsible for displaying inference results for an experiment ID
    from a specific inference run.
    """

    def __init__(self, run_id: str, experiment_id: str):
        """
        Creates a component to display inference results on the screen.

        @param run_id: ID of the inference run.
        @param experiment_id: ID of the experiment to plot.
        """
        self.run_id = run_id
        self.experiment_id = experiment_id
