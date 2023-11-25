import json

import asyncio
import streamlit as st
import os
from collections import OrderedDict


def create_run_page():
    """
    Creates the run page layout.
    """

    st.header("Trigger Inference")
    inference_pane = st.empty()
    _populate_inference_pane(inference_pane)

    st.header("Progress")
    progress_pane = st.empty()
    asyncio.run(_populate_progress_pane(progress_pane, refresh_rate=20))


def _populate_inference_pane(inference_pane: st.container):
    """
    Populates the inference pane where one can trigger a new inference process.

    @param inference_pane: container to place the elements of the pane into.
    """
    with inference_pane:
        with st.container():
            tab1, tab2 = st.columns(2)
            with tab1:
                out_dir = st.text_input(label="out_dir", value=".run")
            with tab2:
                model_params_dict = st.text_area(label="Model Parameters",
                                                 value=json.loads('{"mean_uc0":0}'))
                data_mapping = st.text_area(label="Data Mapping",
                                                 value=json.loads('{"mean_uc0":0}'))
            submit = st.button(label="Submit")
            st.text_area(label="Terminal Output", disabled=True)


async def _populate_progress_pane(progress_pane: st.container, refresh_rate: int):
    """
    Populates the progress pane where one can see the progress of the different inference
    processes.

    @param progress_pane: container to place the elements of the pane into.
    @param refresh_rate: number of seconds to wait before refreshing the pane.
    """
    while True:
        with progress_pane:
            with st.container():
                status_text = st.empty()
                run_ids = [run_id for run_id in sorted(os.listdir(".run"), reverse=True) if
                           os.path.isdir(f".run/{run_id}")]
                for i, run_id in enumerate(run_ids):
                    with open(f".run/{run_id}/execution_params.json", "r") as f:
                        execution_params = json.load(f)
                    total_samples_per_chain = execution_params["burn_in"] + execution_params[
                        "num_samples"]

                    with st.expander(run_id, expanded=(i == 0)):
                        st.json(execution_params, expanded=False)
                        for experiment_id in os.listdir(f".run/{run_id}"):
                            if not os.path.isdir(f".run/{run_id}/{experiment_id}"):
                                continue

                            with open(f".run/{run_id}/{experiment_id}/progress.json", "r") as f:
                                progress = json.load(f)

                            st.write(f"## {experiment_id}")
                            for key, value in OrderedDict(progress["samples"]).items():
                                perc_value = value / total_samples_per_chain
                                text = f"{key} - {value} out of {total_samples_per_chain} - " \
                                       f"{100.0 * perc_value}%"
                                st.progress(perc_value, text=text)

                for i in range(refresh_rate, 0, -1):
                    status_text.write(f"**Refreshing in :red[{i} seconds].**")
                    await asyncio.sleep(1)
