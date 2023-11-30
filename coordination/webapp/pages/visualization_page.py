from typing import List

import numpy as np
import plotly.figure_factory as ff
import streamlit as st

from coordination.inference.inference_data import InferenceData
from coordination.webapp.utils import (create_dropdown_with_default_selection,
                                       get_execution_params,
                                       get_inference_run_ids, plot_curve)


def create_visualization_page():
    run_id = create_dropdown_with_default_selection(
        label="Inference run ID",
        key="inference_run_id_single_visualization",
        values=get_inference_run_ids(),
    )

    if not run_id:
        return

    execution_params_dict = get_execution_params(run_id)
    selected_experiment_ids = st.multiselect(
        "Experiment IDs", options=execution_params_dict["experiment_ids"]
    )

    # Tab 1 contains coordination plots and convergence data.
    # Tab 2 contains plots for any variable the user wants to see.
    tab1, tab2 = st.columns(2)
    with tab1:
        st.write("## Coordination")

    with tab2:
        st.write("## Model variable")
        model_variable = create_dropdown_with_default_selection(
            label="Variable",
            key="model_variable_visualization",
            values=get_inference_run_ids(),
        )

    st.divider()

    tab1, tab2 = st.columns(2)
    if execution_params_dict:
        with tab1:
            _populate_coordination_pane(
                run_id=run_id, experiment_ids=selected_experiment_ids
            )
        with tab2:
            _populate_model_variable_pane(
                run_id=run_id, experiment_ids=selected_experiment_ids
            )


def _populate_coordination_pane(run_id: str, experiment_ids: List[str]):
    """
    Populates pane with a list of coordination plots and convergence table from a list of
    experiments.

    @param run_id: ID of the inference run.
    @param experiment_ids: list of experiment IDs in the run to analyze.
    """
    inference_dir = st.session_state["inference_results_dir"]
    for experiment_id in experiment_ids:
        st.write(f"### {experiment_id}")

        experiment_dir = f"{inference_dir}/{run_id}/{experiment_id}"
        idata = InferenceData.from_trace_file_in_directory(experiment_dir)

        if idata:
            means, stds = idata.average_samples("coordination", return_std=True)
        else:
            st.write(":red[No inference data found.]")
            continue

        plot_curve(
            means=means,
            stds=stds,
            margin_settings=dict(
                l=0,  # Adjust the left margin
                r=0,  # Adjust the right margin
                b=50,  # Adjust the bottom margin
                t=0,  # Adjust the top margin
            ),
        )

        if st.checkbox("Show stats", key=f"check_stats_coordination_{experiment_id}"):
            st.write("#### Stats")
            st.write(f"Average mean: {means.mean():.4f}")
            st.write(f"Average median: {np.median(means):.4f}")
            st.write(f"Average std: {means.std():.4f}")
            st.write("Convergence:")
            st.write(idata.generate_convergence_summary())

            fig = ff.create_distplot(
                [means],
                bin_size=0.01,
                show_rug=False,
                group_labels=["Coordination"],
                colors=["rgb(76, 111, 237)"],
            )
            fig.update_layout(title_text="Distribution of mean coordination")
            fig.update_layout()
            st.plotly_chart(fig, use_container_width=True)


def _populate_model_variable_pane(run_id: str, experiment_ids: List[str]):
    for experiment_id in experiment_ids:
        st.write(f"### {experiment_id}")
