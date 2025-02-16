import streamlit as st

from coordination.webapp.component.experiment_id_multi_selection import \
    ExperimentIDMultiSelection
from coordination.webapp.component.inference_results import InferenceResults
from coordination.webapp.component.inference_run_selection import \
    InferenceRunSelection
from coordination.webapp.component.model_variable_selection import \
    ModelVariableSelection
from coordination.webapp.constants import INFERENCE_RESULTS_DIR_STATE_KEY


class SingleRun:
    """
    This class represents a page to analyze a single inference run. It is comprised of two columns
    where one can compare two different variables of an experiment side by side.
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
        inference_run_component = InferenceRunSelection(
            component_key=f"{self.page_key}_inference_run_selector",
            inference_dir=st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY],
        )
        inference_run_component.create_component()

        if not inference_run_component.selected_inference_run_:
            # Do not render anything below this until an inference run is selected.
            return

        experiment_ids_component = ExperimentIDMultiSelection(
            component_key=f"{self.page_key}_experiments_selector",
            all_experiment_ids=inference_run_component.selected_inference_run_.experiment_ids,
        )
        experiment_ids_component.create_component()

        col_left, col_right = st.columns(2)
        with col_left:
            model_variable_component_left = ModelVariableSelection(
                component_key=f"{self.page_key}_left_col_model_variable_selector",
                inference_run=inference_run_component.selected_inference_run_,
            )
            model_variable_component_left.create_component()

        with col_right:
            model_variable_component_right = ModelVariableSelection(
                component_key=f"{self.page_key}_right_col_model_variable_selector",
                inference_run=inference_run_component.selected_inference_run_,
            )
            model_variable_component_right.create_component()

        for experiment_id in experiment_ids_component.selected_experiment_ids_:
            # Create columns for each experiment to align the results. Different variables can
            # have results with different sizes on the screen.
            col_left, col_right = st.columns(2)
            with col_left:
                inference_results_component = InferenceResults(
                    component_key=f"{self.page_key}_left_col_{experiment_id}_inference_results",
                    inference_run=inference_run_component.selected_inference_run_,
                    experiment_id=experiment_id,
                    model_variable_info=model_variable_component_left.selected_model_variable_,
                    model_variable_dimension=(
                        model_variable_component_left.selected_dimension_name_
                    ),
                )
                inference_results_component.create_component()

            with col_right:
                inference_results_component = InferenceResults(
                    component_key=f"{self.page_key}_right_col_{experiment_id}_inference_results",
                    inference_run=inference_run_component.selected_inference_run_,
                    experiment_id=experiment_id,
                    model_variable_info=model_variable_component_right.selected_model_variable_,
                    model_variable_dimension=(
                        model_variable_component_right.selected_dimension_name_
                    ),
                )
                inference_results_component.create_component()
