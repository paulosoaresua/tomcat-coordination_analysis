import uuid

import numpy as np
import pandas as pd
import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown
from coordination.webapp.entity.inference_run import InferenceRun
from coordination.webapp.entity.model_variable import ModelVariableInfo
from coordination.inference.inference_data import InferenceData
from coordination.webapp.constants import DEFAULT_COLOR_PALETTE, DEFAULT_PLOT_MARGINS
import itertools
import plotly.figure_factory as ff
import plotly.graph_objects as go


class InferenceStatsComponent:
    """
    Represents a component that displays coordination statistics for an inference.
    """

    def __init__(self, component_key: str, inference_data: InferenceData,
                 convergence_report: pd.DataFrame):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_data: object containing results on an inference.
        @param convergence_report: a convergence report for the inference. We can call this
            directly from the idata object, but we have it here as a parameter so we can cache it
            using the streamlit @st.cache_data annotation.
        """
        self.component_key = component_key
        self.inference_data = inference_data
        self.convergence_report = convergence_report

    def create_component(self):
        """
        Displays coordination statistics.
        """
        if not self.inference_data:
            return

        means = self.inference_data.average_posterior_samples("coordination", return_std=False)
        means = means.to_numpy()

        st.write("#### Coordination stats")
        st.write(
            "*:blue[Statistics computed over the mean posterior coordination per time step.]*")
        st.write(f"Mean: {means.mean():.4f}")
        st.write(f"Median: {np.median(means):.4f}")
        st.write(f"Std: {means.std():.4f}")

        self._plot_coordination_distribution(means)

        st.write("#### Model stats")
        st.write("**Convergence**")
        st.dataframe(self.convergence_report, se_container_width=True)

        self._plot_log_probability_distribution()

    def _plot_coordination_distribution(self, overall_coordination: np.ndarray):
        """
        Plots histogram with the distribution of coordination per chain and combined.

        @param overall_coordination: coordination series averaged across all chains and draws.
        """
        color_palette_iter = itertools.cycle(DEFAULT_COLOR_PALETTE)
        # chain x time
        coordination_per_chain = self.inference_data.trace["posterior"]["coordination"].mean(
            dim=["draw"]).to_numpy()

        # Add combination of all chains
        coordination = np.concatenate([overall_coordination[None, :], coordination_per_chain],
                                      axis=0)
        colors = [next(color_palette_iter) for _ in range(coordination.shape[0])]
        labels = ["All chains"] + [f"Chain {i + 1}" for i in range(coordination.shape[0] - 1)]
        fig = ff.create_distplot(
            coordination,
            bin_size=0.01,
            show_rug=False,
            group_labels=labels,
            colors=colors
        )
        fig.update_layout(title_text="Coordination distribution",
                          xaxis_title="Coordination",
                          yaxis_title="Density",
                          # Preserve legend order
                          legend={"traceorder": "normal"},
                          margin=DEFAULT_PLOT_MARGINS)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_log_probability_distribution(self):
        """
        Plots box plots with the distribution of log-probabilities per chain and combined.
        """
        color_palette_iter = itertools.cycle(DEFAULT_COLOR_PALETTE)
        log_probabilities = self.inference_data.get_log_probs()  # chain x draw
        # Add combination of all chains
        log_probabilities = np.concatenate(
            [np.mean(log_probabilities, axis=0, keepdims=True), log_probabilities], axis=0)
        labels = ["All chains"] + [f"Chain {i + 1}" for i in range(log_probabilities.shape[0] - 1)]
        fig = go.Figure()
        for i in range(log_probabilities.shape[0]):
            color = next(color_palette_iter)
            fig.add_trace(
                go.Box(y=log_probabilities[i],
                       name=labels[i],
                       fillcolor=color,
                       line=dict(color="black"))
            )
        fig.update_layout(title_text="Distribution of log-probabilities",
                          xaxis_title="Log-probability",
                          yaxis_title="Density",
                          # Preserve legend order
                          legend={"traceorder": "normal"},
                          margin=DEFAULT_PLOT_MARGINS)
        st.plotly_chart(fig, use_container_width=True)
