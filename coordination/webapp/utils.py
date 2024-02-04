import itertools
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objs as go
import streamlit as st
import xarray

from coordination.inference.inference_data import InferenceData
from coordination.webapp.constants import DEFAULT_COLOR_PALETTE


def disable_sidebar():
    """
    Removes button to collapse/expand sidebar via CSS.
    """
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def get_inference_run_ids(inference_dir: str) -> List[str]:
    """
    Gets a list of inference run IDs from the list of directories under an inference folder.

    @param inference_dir: directory where inference runs were saved.
    @return: list of inference run ids.
    """
    if os.path.exists(inference_dir):
        run_ids = [
            run_id
            for run_id in os.listdir(inference_dir)
            if os.path.isdir(f"{inference_dir}/{run_id}")
        ]

        # Display on the screen from the most recent to the oldest.
        return sorted(run_ids, reverse=True)

    return []


class DropDownOption:
    """
    This class represents a dropdown option with an optional text prefix.
    """

    def __init__(self, name: str, prefix: Optional[str] = None):
        """
        Creates a dropdown option.

        @param name: option name.
        @param prefix: prefix to be prepended to the option.
        """
        self.prefix = prefix
        self.name = name

    def __repr__(self) -> str:
        """
        Gets a textual representation of the option with a prepended prefix if not undefined.

        @return: textual representation of the option.
        """
        if self.prefix:
            return f"{self.prefix} {self.name}"
        else:
            return self.name


def create_dropdown_with_default_selection(
        label: str, key: str, options: List[DropDownOption]
) -> Optional[str]:
    """
    Creates a dropdown with an extra value for default selection.

    @param label: label of the dropdown.
    @param key: key of the dropdown.
    @param options: values to be selected.
    @return: value selected in the dropdown.
    """

    def format_func(x: DropDownOption):
        """
        Define which text to display for an option in a dropdown.

        @param x: option of the list.
        @return: text for the option to be displayed in the dropdown.
        """
        if x:
            return str(x)
        else:
            return "-- Select a value --"

    value = st.selectbox(
        label,
        key=key,
        options=[None] + options,
        # If it's a tuple, add the first item as prefix of the value name in the second item.
        format_func=format_func,
    )

    if isinstance(value, tuple):
        # Return the name of the variable.
        return value[1]

    return value


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


def get_model_variables(run_id: str) -> Dict[str, List[str]]:
    """
    Gets a dictionary with a list of tuples containing model variable names and associated list of
    dimension coordinates from the inference data available for any experiment id in an inference
    run. We divide the lists into latent, observed, prior and posterior predictive variables, which
    determines the keys of the dictionary.

    @param run_id: ID of the inference run.
    @return: dictionary with list of variables sampled during inference.
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
            # Found an idata. That suffices to get the list of variables since the model is the
            # same per run ID.
            break

    if not idata:
        return []

    latent_variable_names = idata.posterior_latent_data_variables
    latent_parameter_variable_names = idata.posterior_latent_parameter_variables
    observed_variable_names = idata.observed_data_variables
    prior_predictive = idata.prior_predictive_variables
    posterior_predictive = idata.posterior_predictive_variables

    return {
        "latent": [
            (var_name, idata.get_dimension_coordinates(var_name))
            for var_name in latent_variable_names
        ],
        "observed": [
            (var_name, idata.get_dimension_coordinates(var_name))
            for var_name in observed_variable_names
        ],
        "latent_parameter": [
            (var_name, idata.get_dimension_coordinates(var_name))
            for var_name in latent_parameter_variable_names
        ],
        "prior_predictive": [
            (var_name, idata.get_dimension_coordinates(var_name))
            for var_name in prior_predictive
        ],
        "posterior_predictive": [
            (var_name, idata.get_dimension_coordinates(var_name))
            for var_name in posterior_predictive
        ],
    }


def plot_curve(
        variable_name: str,
        inference_data: InferenceData,
        inference_mode: str,
        dimension: Union[int, str] = 0,
) -> go.Figure:
    """
    Plots the time series of samples drawn from the posterior distribution.

    @param variable_name: variable to plot.
    @param inference_data: object with the samples.
    @param inference_mode: one of prior_check, posterior or posterior_predictive.
    @param dimension: index or name of the dimension axis to plot.
    @raise ValueError: if the mode was not found in the inference data.
    @return: a figure with new plots added to it.
    """

    color_palette_iterator = itertools.cycle(DEFAULT_COLOR_PALETTE)
    if inference_mode == "posterior":
        means, stds = inference_data.average_posterior_samples(
            variable_name, return_std=True
        )
    elif inference_mode == "prior_predictive":
        means, stds = inference_data.average_prior_predictive_samples(
            variable_name, return_std=True
        )
    elif inference_mode == "posterior_predictive":
        means, stds = inference_data.average_posterior_predictive_samples(
            variable_name, return_std=True
        )
    else:
        raise ValueError(f"Invalid inference mode ({inference_mode}).")

    if len(means.shape) == 1:
        # The series only has a time axis.
        time_steps = np.arange(len(means))

        bounds = [0, 1] if variable_name == "coordination" else None
        fig = plot_series(
            x=time_steps,
            y=means,
            y_std=stds,
            value_bounds=bounds,
            color=next(color_palette_iterator),
        )

        yaxis_title = variable_name
    else:  # len(means.shape) == 2:
        # Serial variable: the first axis is the dimension of the latent component.
        subject_indices = np.array(
            [int(x.split("#")[0]) for x in getattr(means, f"{variable_name}_time").data]
        )
        time_steps = np.array(
            [int(x.split("#")[1]) for x in getattr(means, f"{variable_name}_time").data]
        )
        subjects = sorted(list(set(subject_indices)))
        fig = None
        for s in subjects:
            idx = [i for i, subject in enumerate(subject_indices) if subject == s]
            y = (
                means.loc[dimension][idx]
                if isinstance(dimension, str)
                else means[dimension, idx]
            )
            if stds is None:
                y_std = None
            else:
                y_std = (
                    stds.loc[dimension][idx]
                    if isinstance(dimension, str)
                    else stds[dimension, idx]
                )

            fig = plot_series(
                x=time_steps[idx],
                y=y,
                y_std=y_std,
                label=f"Subject {s}",
                figure=fig,
                color=next(color_palette_iterator),
            )

        yaxis_title = f"{variable_name} - {dimension}"

    fig.update_layout(xaxis_title="Time Step", yaxis_title=yaxis_title)

    st.plotly_chart(fig, use_container_width=True)

    return fig


def plot_series(
        x: Union[np.ndarray, xarray.DataArray],
        y: Union[np.ndarray, xarray.DataArray],
        y_std: Union[np.ndarray, xarray.DataArray] = None,
        label: Optional[str] = None,
        value_bounds: Optional[Tuple[float, float]] = None,
        figure: Optional[go.Figure] = None,
        color: Optional[str] = None,
        marker: Optional[bool] = False
) -> go.Figure:
    """
    Plots a time series with optional error bands.

    @param x: time steps.
    @param y: values for each time step.
    @param y_std: optional standard deviation of the values for error band plotting.
    @param label: an optional label for the series. If undefined, legends won't be shown.
    @param value_bounds: optional tuple containing min and max values to be plotted. Values out of
        the range will be capped.
    @param figure: an optional existing figure to add curves to. If not provided, a new figure
        will be created.
    @param color: an optional color for the line and error bands.
    @param marker: whether to put marker on points.
    @return: a figure with new plots added to it.
    """

    if not figure:
        if (
                value_bounds is not None
                and value_bounds[0] is not None
                and value_bounds[1] is not None
        ):
            figure = go.Figure(layout_yaxis_range=value_bounds)
        else:
            figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers" if marker else "lines",
            line=dict(color=color),
            name=label,
            showlegend=label is not None
        )
    )

    # Error bands
    if y_std is not None:
        lower_band = y - y_std
        upper_band = y + y_std

        if value_bounds and value_bounds[0]:
            lower_band = np.maximum(lower_band, value_bounds[0])

        if value_bounds and value_bounds[1]:
            upper_band = np.minimum(upper_band, value_bounds[1])

        faded_color = f"rgba{color[3:-1]}, {0.2})" if color else None
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([x, np.flip(x)]),
                y=np.concatenate([upper_band, np.flip(lower_band)]),
                fill="toself",
                fillcolor=faded_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"{label} - band" if label else "band",
                showlegend=True,
            ),
        )

    return figure
