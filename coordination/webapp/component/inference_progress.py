import asyncio
from typing import List, Optional

import streamlit as st

from coordination.inference.inference_run import InferenceRun
from coordination.webapp.component.inference_run_progress import \
    InferenceRunProgress
from coordination.webapp.constants import DATA_DIR_STATE_KEY
from coordination.webapp.utils import get_inference_run_ids


class InferenceProgress:
    """
    Represents a component that displays a collection of inference runs and the progress of each
    one of them.
    """

    def __init__(
        self,
        component_key: str,
        inference_dir: str,
        refresh_rate: int,
        preferred_run_ids: Optional[List[str]] = None,
        display_experiment_progress: bool = True,
        display_sub_experiment_progress: bool = True,
        hide_completed_experiment: bool = True,
    ):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_dir: directory where inference runs were saved.
        @param refresh_rate: how many seconds to wait before updating the progress.
        @param preferred_run_ids: a collection of run ids to show the progress. If not provided,
            the progress of all run ids in the inference directory will be displayed.
        @param display_experiment_progress: whether to display the progress of all the experiments
            in the inference run.
        @param display_sub_experiment_progress: whether to display the progress of all the
            sub-experiments of all the experiments in the inference run.
        @param hide_completed_experiment: whether to hide successfully completed experiments from
            the list.
        """
        self.component_key = component_key
        self.inference_dir = inference_dir
        self.preferred_run_ids = preferred_run_ids
        self.refresh_rate = refresh_rate
        self.display_experiment_progress = display_experiment_progress
        self.display_sub_experiment_progress = display_sub_experiment_progress
        self.hide_completed_experiment = hide_completed_experiment

    def create_component(self):
        """
        Creates area in the screen for selection of an inference run id. Below is presented a json
        object with the execution params of the run once one is chosen from the list.
        """
        asyncio.run(self._create_progress_area())

    async def _create_progress_area(self):
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
                    if self.preferred_run_ids:
                        run_ids = self.preferred_run_ids
                    else:
                        run_ids = get_inference_run_ids(self.inference_dir)

                    # The status contains a countdown showing how many seconds until the next
                    # refresh. It is properly filled in the end of this function after we parse
                    # all the experiments in the run and know how many of them have finished
                    # successfully.
                    status_text = st.empty()

                    if len(run_ids) <= 0:
                        await self._wait(status_text)
                        continue

                    for i, run_id in enumerate(run_ids):
                        inference_run = InferenceRun(
                            inference_dir=self.inference_dir,
                            run_id=run_id,
                            data_dir=st.session_state[DATA_DIR_STATE_KEY],
                        )

                        if not inference_run.execution_params:
                            continue

                        # Pre-expand just the first run in the list
                        with st.expander(run_id, expanded=(i == 0)):
                            inference_progress_component = InferenceRunProgress(
                                inference_run,
                                self.display_experiment_progress,
                                self.display_sub_experiment_progress,
                                self.hide_completed_experiment,
                            )
                            inference_progress_component.create_component()

                    await self._wait(status_text)

    async def _wait(self, countdown_area: st.container):
        """
        Waits a few seconds and update countdown.
        """
        for i in range(self.refresh_rate, 0, -1):
            countdown_area.write(f"**Refreshing in :red[{i} seconds].**")
            await asyncio.sleep(1)
