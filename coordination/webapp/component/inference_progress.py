import asyncio

import streamlit as st

from coordination.webapp.component.inference_run_progress import \
    InferenceRunProgress
from coordination.webapp.entity.inference_run import InferenceRun
from coordination.webapp.utils import get_inference_run_ids


class InferenceProgress:
    """
    Represents a component that displays a collection of inference runs and the progress of each
    one of them.
    """

    def __init__(self, component_key: str, inference_dir: str, refresh_rate: int):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_dir: directory where inference runs were saved.
        @param refresh_rate: how many seconds to wait before updating the progress.
        """
        self.component_key = component_key
        self.inference_dir = inference_dir
        self.refresh_rate = refresh_rate

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """
        self._create_progress_area()

    def _create_progress_area(self):
        """
        Populates the progress pane where one can see the progress of the different inference runs.

        WARNING:
        It's not possible to have widgets that require unique keys in this pane because the widget
        keys are not cleared until the next run. We could keep creating different keys but this
        would cause memory leakage as the keys would be accumulated in the run context.
        """
        progress_area = st.empty()
        while True:
            with progress_area:
                with st.container():
                    run_ids = get_inference_run_ids(self.inference_dir)

                    for i, run_id in enumerate(run_ids):
                        inference_run = InferenceRun(
                            inference_dir=self.inference_dir, run_id=run_id
                        )

                        if not inference_run.execution_params:
                            continue

                        # Pre-expand just the first run in the list
                        with st.expander(run_id, expanded=(i == 0)):
                            inference_progress_component = InferenceRunProgress(
                                inference_run
                            )
                            inference_progress_component.create_component()
