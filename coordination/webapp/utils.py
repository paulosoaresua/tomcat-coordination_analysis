import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objs as go
import streamlit as st

from coordination.webapp.constants import INFERENCE_PARAMETERS_DIR


def get_saved_execution_parameter_files() -> List[str]:
    """
    Gets the list of files with saved execution parameters.

    @return: list of files with saved execution parameters.
    """
    if os.path.exists(INFERENCE_PARAMETERS_DIR):
        saved_params_list = sorted(
            [
                f
                for f in os.listdir(INFERENCE_PARAMETERS_DIR)
                if os.path.isfile(f"{INFERENCE_PARAMETERS_DIR}/{f}")
            ]
        )
        return saved_params_list

    return []


def create_dropdown_with_default_selection(
    label: str, key: str, values: List[str]
) -> Optional[str]:
    """
    Creates a dropdown with an extra value for default selection.

    @param label: label of the dropdown.
    @param key: key of the dropdown.
    @param values: values to be selected.
    @return: value selected in the dropdown.
    """
    return st.selectbox(
        label,
        key=key,
        options=[None] + values,
        format_func=lambda x: x if x else "-- Select a value --",
    )


def get_inference_run_ids() -> List[str]:
    """
    Gets a list of inference run IDs from the list of directories under an inference folder.

    @return: list of inference run ids.
    """
    inference_dir = st.session_state["inference_results_dir"]
    if os.path.exists(inference_dir):
        run_ids = [
            run_id
            for run_id in sorted(os.listdir(inference_dir), reverse=True)
            if os.path.isdir(f"{inference_dir}/{run_id}")
        ]
        return run_ids

    return []


def get_execution_params(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Gets the json object containing execution parameters for a specific inference run.

    @param run_id: ID of the run.
    @return: list of experiment IDs processed in a run ID.
    """
    inference_dir = st.session_state["inference_results_dir"]
    execution_params_filepath = f"{inference_dir}/{run_id}/execution_params.json"
    if not os.path.exists(execution_params_filepath):
        return None

    with open(execution_params_filepath, "r") as f:
        execution_params_dict = json.load(f)

    return execution_params_dict


def plot_curve(
    means: np.ndarray,
    stds: Optional[np.ndarray],
    margin_settings: Optional[Dict[str, float]],
):
    lower_band = means - stds
    upper_band = means + stds

    time_stamps = np.arange(len(means))
    fig = go.Figure(
        [
            go.Scatter(
                x=time_stamps,
                y=means,
                line=dict(color="rgb(76,111,237)"),
                mode="lines",
                showlegend=False,
            ),
            # Bands
            go.Scatter(
                x=np.concatenate([time_stamps, np.flip(time_stamps)]),
                y=np.concatenate([upper_band, np.flip(lower_band)]),
                fill="toself",
                fillcolor="rgba(76,111,237,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
        ]
    )

    fig.update_layout(margin=margin_settings)
    st.plotly_chart(fig, use_container_width=True)
