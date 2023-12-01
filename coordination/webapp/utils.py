import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objs as go
import streamlit as st

from coordination.webapp.constants import INFERENCE_PARAMETERS_DIR
from coordination.inference.inference_data import InferenceData


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
        label: str, key: str, values: List[Union[str, Tuple[str, str]]]
) -> Optional[str]:
    """
    Creates a dropdown with an extra value for default selection.

    @param label: label of the dropdown.
    @param key: key of the dropdown.
    @param values: values to be selected.
    @return: value selected in the dropdown.
    """

    def format_func(x: Union[str, Tuple[str, str]]):
        """
        Define which text to display for an option in a dropdown.

        @param x: option of the list.
        @return: text for the option to be displayed in the dropdown.
        """
        if x:
            if isinstance(x, tuple):
                # If the option is a tuple, use the the first item as prefix of the value name in
                # the second item.
                return f"{x[0]} {x[1]}"
            else:
                return x
        else:
            return "-- Select a value --"

    value = st.selectbox(
        label,
        key=key,
        options=[None] + values,
        # If it's a tuple, add the first item as prefix of the value name in the second item.
        format_func=format_func
    )

    if isinstance(value, tuple):
        # Return the name of the variable.
        return value[1]

    return value


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
        variable_name: str,
        inference_data: InferenceData,
        dimension_idx: int = 0,
        margin_settings: Optional[Dict[str, float]] = None,
):
    means, stds = inference_data.average_samples(variable_name, return_std=True)

    lower_band = means - stds
    upper_band = means + stds

    plot_objects = []

    if len(means.shape) == 1:
        # The series only has a time axis.
        time_stamps = np.arange(len(means))
        plot_objects.append(
            go.Scatter(
                x=time_stamps,
                y=means,
                line=dict(color="rgb(76,111,237)"),
                mode="lines",
                showlegend=False,
            )
        )

        # Bands
        plot_objects.append(
            go.Scatter(
                x=np.concatenate([time_stamps, np.flip(time_stamps)]),
                y=np.concatenate([upper_band, np.flip(lower_band)]),
                fill="toself",
                fillcolor="rgba(76,111,237,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
        )
    elif len(means.shape) == 2:
        # Serial variable: the first axis is the dimension of the latent component.
        subject_indices = np.array(
            [
                int(x.split("#")[0])
                for x in getattr(means, f"{variable_name}_time").data
            ]
        )
        time_steps = np.array(
            [
                int(x.split("#")[1])
                for x in getattr(means, f"{variable_name}_time").data
            ]
        )
        subjects = sorted(list(set(subject_indices)))
        for s in subjects:
            idx = [i for i, subject in enumerate(subject_indices) if subject == s]
            plot_objects.append(
                go.Scatter(
                    x=time_steps[idx],
                    y=means[dimension_idx, idx],
                    # line=dict(color="rgb(76,111,237)"),
                    mode="lines",
                    name=f"Subject {s}",
                    showlegend=True,
                )
            )

    fig = go.Figure(plot_objects)
    fig.update_layout(margin=margin_settings)
    st.plotly_chart(fig, use_container_width=True)


def get_model_variables(run_id: str) -> Dict[str, List[str]]:
    """
    Gets a dictionary with a list of model variable names from the inference data available for
    any experiment id in an inference run. We divide the lists into "latent" and "observed"
    variables. They are indexed by the keys "latent" and "observed" in the dictionary respectively.

    @param run_id: ID of the inference run.
    @return: list of model variables.
    """
    execution_params_dict = get_execution_params(run_id)

    if not execution_params_dict:
        return []

    idata = None
    inference_dir = st.session_state["inference_results_dir"]
    for experiment_id in execution_params_dict["experiment_ids"]:
        experiment_dir = f"{inference_dir}/{run_id}/{experiment_id}"
        idata = InferenceData.from_trace_file_in_directory(experiment_dir)
        if idata:
            # Found one idata. That suffices to get the list of variables since the model is the
            # same per run ID.
            break

    if not idata:
        return []

    return {"latent": idata.posterior_latent_variables,
            "observed": idata.observed_variables}
