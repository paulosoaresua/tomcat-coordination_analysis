import streamlit as st

from coordination.webapp.component.experiment_id_multi_selection import \
    ExperimentIDMultiSelection
from coordination.webapp.component.evaluation_results import EvaluationResults
from coordination.webapp.component.inference_run_selection import \
    InferenceRunSelection
from coordination.webapp.component.model_variable_selection import \
    ModelVariableSelection
from coordination.webapp.constants import INFERENCE_RESULTS_DIR_STATE_KEY
from coordination.webapp.constants import EVALUATIONS_DIR


class RunVsRunEvaluations:
    """
    This class represents a page to compare two evaluations from two inference runs side by side.
    """

    def __init__(self, page_key: str):
        """
        Creates the page object.

        @param page_key: unique identifier for the page. Components of the page will append to
            this key to form their keys.
        """
        self.page_key = page_key

    def create_page(self):
        """
        Creates the page by adding different components to the screen.
        """
        st.write(
            f"Evaluations dir: *:blue[{EVALUATIONS_DIR}]*"
        )

        col_left, col_right = st.columns(2)
        with col_left:
            RunVsRunEvaluations._populate_column(column_key=f"{self.page_key}_left_col")

        with col_right:
            RunVsRunEvaluations._populate_column(column_key=f"{self.page_key}_right_col")

    @staticmethod
    def _populate_column(column_key: str):
        """
        Populates a column where one can choose an inference run.

        @param column_key: unique identifier for the column. Components of the column will append
            to this key to form their keys.
        """
        inference_run_component = InferenceRunSelection(
            component_key=f"{column_key}_inference_run_selector",
            inference_dir=EVALUATIONS_DIR,
        )
        inference_run_component.create_component()

        if not inference_run_component.selected_inference_run_:
            # Do not render anything below this until an inference run is selected.
            return

        evaluation_results_component = EvaluationResults(
            component_key=f"{column_key}_{inference_run_component.selected_inference_run_.run_id}"
                          f"_evaluation_results",
            inference_run=inference_run_component.selected_inference_run_
        )
        evaluation_results_component.create_component()
