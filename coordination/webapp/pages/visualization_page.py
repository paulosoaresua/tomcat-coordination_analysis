from typing import List, Optional

import numpy as np
import plotly.figure_factory as ff
import streamlit as st

from coordination.inference.inference_data import InferenceData
from coordination.webapp.utils import (create_dropdown_with_default_selection,
                                       get_execution_params,
                                       get_inference_run_ids, plot_curve, get_model_variables)

COORDINATION_STATS_VARIABLE = "coordination_stats"
LATENT_PARAMETERS_VARIABLE = "latent_parameters"


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

    # Each tab can contain information about one variable of the model for quick comparison
    # Two extra variables are added: one for coordination stats and another one for parameter
    # plots.
    # Get a sample idata to see what variables are available as latent and observation.
    model_variables = get_model_variables(run_id)
    variable_names = [("[LATENT]", v) for v in model_variables["latent"]]
    variable_names += [("[OBSERVED]", v) for v in model_variables["observed"]]
    variable_names += [("[EXTRA]", COORDINATION_STATS_VARIABLE),
                       ("[EXTRA]", LATENT_PARAMETERS_VARIABLE)]
    variable_names.sort()
    tab1, tab2 = st.columns(2)
    with tab1:
        st.write("## Model variable")
        model_variable_left = create_dropdown_with_default_selection(
            label="Variable",
            key="model_variable_visualization_left",
            values=variable_names
        )

    with tab2:
        st.write("## Model variable")
        model_variable_right = create_dropdown_with_default_selection(
            label="Variable",
            key="model_variable_visualization_right",
            values=variable_names
        )

    st.divider()

    if execution_params_dict:
        for experiment_id in selected_experiment_ids:
            tab1, tab2 = st.columns(2)
            with tab1:
                _populate_variable_pane(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    variable_name=model_variable_left
                )
            with tab2:
                _populate_variable_pane(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    variable_name=model_variable_right
                )


def _populate_variable_pane(run_id: str, experiment_id: str, variable_name: Optional[str]):
    """
    Populates pane with a list of coordination plots and convergence table from a list of
    experiments.

    @param run_id: ID of the inference run.
    @param experiment_id: experiment ID in the run to analyze.
    """
    if not variable_name:
        return

    inference_dir = st.session_state["inference_results_dir"]
    st.write(f"### {experiment_id}")

    experiment_dir = f"{inference_dir}/{run_id}/{experiment_id}"
    idata = InferenceData.from_trace_file_in_directory(experiment_dir)

    if not idata:
        st.write(":red[No inference data found.]")
        return

    if variable_name == COORDINATION_STATS_VARIABLE:
        means = idata.average_samples("coordination", return_std=False)
        means = means.to_numpy()

        st.write("#### Stats")
        st.write(f"Average mean: {means.mean():.4f}")
        st.write(f"Average median: {np.median(means):.4f}")
        st.write(f"Std. of the mean: {means.std():.4f}")
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
    elif variable_name == LATENT_PARAMETERS_VARIABLE:
        st.pyplot(idata.plot_parameter_posterior(), clear_figure=True)
    else:
        plot_curve(
            variable_name=variable_name,
            inference_data=idata,
            margin_settings=dict(
                l=0,  # Adjust the left margin
                r=0,  # Adjust the right margin
                b=50,  # Adjust the bottom margin
                t=0,  # Adjust the top margin
            ),
        )
