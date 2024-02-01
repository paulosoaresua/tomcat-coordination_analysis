import itertools
from typing import Optional

import numpy as np
import plotly.figure_factory as ff
import streamlit as st

from coordination.inference.inference_data import InferenceData
from coordination.inference.model_variable import ModelVariableInfo
from coordination.webapp.constants import (DEFAULT_COLOR_PALETTE,
                                           DEFAULT_PLOT_MARGINS)
from coordination.webapp.utils import plot_series


class ModelVariableInferenceResults:
    """
    Represents a component that displays inference results for a model variable in an experiment
    from a particular inference run.
    """

    def __init__(
        self,
        component_key: str,
        model_variable_info: ModelVariableInfo,
        dimension: Optional[str],
        inference_data: InferenceData,
    ):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param model_variable_info: object containing info about the model variable.
        @param dimension: dimension if the variable has more than one to choose from.
        @param inference_data: object containing results on an inference.

        """
        self.component_key = component_key
        self.model_variable_info = model_variable_info
        self.dimension = dimension
        self.inference_data = inference_data

    def create_component(self):
        """
        Displays inference results for a specific model variable.
        """
        if not self.model_variable_info:
            return

        if not self.inference_data:
            return

        if self.inference_data.is_parameter(
            self.model_variable_info.inference_mode,
            self.model_variable_info.variable_name,
        ):
            self._display_parameter_variable_histogram()
        else:
            self._display_data_variable_time_series_curve()

    def _display_parameter_variable_histogram(self):
        """
        Plots the histogram of the sampled parameter variable. We add dropdowns for dimension
        selection such that only one dimension is plotted at a time.
        """
        # Axes = (chain, draw, dim1, dim2...) or (dim1, dim2, ...) if inference mode is
        # observed_data
        samples = self.inference_data.trace[self.model_variable_info.inference_mode][
            self.model_variable_info.variable_name
        ]

        if self.model_variable_info.inference_mode == "observed_data":
            # A single tensor value
            st.write(f"{self.model_variable_info.variable_name} = {samples.to_numpy()}")
        else:
            # Add dropdowns to choose the dimensions to index such that we only produce one
            # histogram per chain at a time.
            dim_indices = []
            # The first two axis are chain and draw.
            for i, num_dimensions_in_axis in enumerate(samples.shape[2:]):
                dim_indices.append(
                    st.selectbox(
                        f"Dimension {i + 1}",
                        key=f"{self.component_key}_parameter_variable_dimension_selector_{i}",
                        options=range(num_dimensions_in_axis),
                    )
                )

            if len([idx for idx in dim_indices if idx is None]) == 0:
                # All dimensions selected
                for dim_idx in dim_indices:
                    samples = samples[:, :, dim_idx]

                samples = samples.to_numpy()
                # Add the average across all chains to the list
                samples = np.concatenate(
                    [np.mean(samples, axis=0, keepdims=True), samples]
                )
                color_palette_iter = itertools.cycle(DEFAULT_COLOR_PALETTE)
                colors = [next(color_palette_iter) for _ in range(samples.shape[0])]
                labels = ["All chains"] + [
                    f"Chain {chain + 1}" for chain in range(samples.shape[0] - 1)
                ]
                fig = ff.create_distplot(
                    samples,
                    # bin_size=0.01,
                    show_rug=False,
                    group_labels=labels,
                    colors=colors,
                )
                fig.update_layout(
                    xaxis_title=self.model_variable_info.variable_name,
                    yaxis_title="Density",
                    # Preserve legend order
                    legend={"traceorder": "normal"},
                    margin=DEFAULT_PLOT_MARGINS,
                )
                st.plotly_chart(fig, use_container_width=True)

    def _display_data_variable_time_series_curve(self):
        """
        Plots the sampled time series of the data variable and corresponding error bands (one
        standard deviation from the sample mean).
        """

        if self.model_variable_info.inference_mode == "observed_data":
            # There's no notion of chain or draw for observed data. That is given, not sampled.
            means = self.inference_data.trace[self.model_variable_info.inference_mode][
                self.model_variable_info.variable_name
            ]
            stds = None
        else:
            means = self.inference_data.trace[self.model_variable_info.inference_mode][
                self.model_variable_info.variable_name
            ].mean(dim=["chain", "draw"])
            stds = self.inference_data.trace[self.model_variable_info.inference_mode][
                self.model_variable_info.variable_name
            ].std(dim=["chain", "draw"])

        fig = None
        color_palette_iter = itertools.cycle(DEFAULT_COLOR_PALETTE)
        if means.ndim == 1:
            # The series only has a time axis.
            time_steps = np.arange(len(means))
            bounds = (
                [0, 1]
                if self.model_variable_info.variable_name == "coordination"
                else None
            )
            fig = plot_series(
                x=time_steps,
                y=means,
                y_std=stds,
                value_bounds=bounds,
                color=next(color_palette_iter),
            )
            yaxis_label = self.model_variable_info.variable_name
        else:  # len(means.shape) == 2:
            # Serial variable: the first axis is the dimension and the second is the time.

            if self.dimension is None:
                return

            if means.ndim == 2:
                # Serialized data. Axes are (dimension x time)
                # Get subject indices and time indices from the coded coordinates of the data along
                # the time axis. Each time coordinate is coded as "subject#time" when a serial
                # module is created.
                subject_indices = np.array(
                    [
                        int(x.split("#")[0])
                        for x in getattr(
                            means, f"{self.model_variable_info.variable_name}_time"
                        ).data
                    ]
                )
                time_steps = np.array(
                    [
                        int(x.split("#")[1])
                        for x in getattr(
                            means, f"{self.model_variable_info.variable_name}_time"
                        ).data
                    ]
                )
                unique_subjects = sorted(list(set(subject_indices)))
                for s in unique_subjects:
                    # Get the indices in the time series belonging to subject "s". In a serial
                    # module, only one subject is observed at a time.
                    idx = [
                        i for i, subject in enumerate(subject_indices) if subject == s
                    ]
                    y = (
                        means.loc[self.dimension][idx]
                        if isinstance(self.dimension, str)
                        else means[self.dimension, idx]
                    )
                    if stds is None:
                        y_std = None
                    else:
                        y_std = (
                            stds.loc[self.dimension][idx]
                            if isinstance(self.dimension, str)
                            else stds[self.dimension, idx]
                        )

                    fig = plot_series(
                        x=time_steps[idx],
                        y=y,
                        y_std=y_std,
                        label=f"Subject {s}",
                        figure=fig,
                        color=next(color_palette_iter),
                    )
            else:
                # Non-serialized data. Axes are (subject x dimension x time)
                for s in range(means.shape[0]):
                    y = (
                        means[s].loc[self.dimension]
                        if isinstance(self.dimension, str)
                        else means[s, self.dimension]
                    )
                    if stds is None:
                        y_std = None
                    else:
                        y_std = (
                            stds[s].loc[self.dimension]
                            if isinstance(self.dimension, str)
                            else stds[s, self.dimension]
                        )

                    fig = plot_series(
                        x=np.arange(len(y)),
                        y=y,
                        y_std=y_std,
                        label=f"Subject {s}",
                        figure=fig,
                        color=next(color_palette_iter),
                    )

            yaxis_label = f"{self.model_variable_info.variable_name} - {self.dimension}"

        if fig:
            fig.update_layout(
                xaxis_title="Time Step",
                yaxis_title=yaxis_label,
                # Preserve legend order
                legend={"traceorder": "normal"},
                margin=DEFAULT_PLOT_MARGINS,
            )

            st.plotly_chart(fig, use_container_width=True)
