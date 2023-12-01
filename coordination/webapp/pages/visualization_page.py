from typing import List, Optional

import numpy as np
import plotly.figure_factory as ff
import streamlit as st

from coordination.inference.inference_data import InferenceData
from coordination.webapp.utils import (create_dropdown_with_default_selection,
                                       get_execution_params,
                                       get_inference_run_ids, plot_curve, get_model_variables)
from coordination.webapp.utils import DropDownOption

COORDINATION_STATS_VARIABLE = "coordination_stats"
LATENT_PARAMETERS_VARIABLE = "latent_parameters"


class DropDownOptionWithInferenceMode(DropDownOption):
    """
    This class represents a dropdown option extended with an extra inference mode parameter.
    """

    def __init__(self, name: str, inference_mode: str, prefix: Optional[str] = None):
        """
        Creates a dropdown option.

        @param name: option name.
        @param inference mode. One of posterior, prior_check or posterior_predictive.
        @param prefix: prefix to be prepended to the option.
        """
        super().__init__(name, prefix)
        self.inference_mode = inference_mode


def create_visualization_page():
    run_id = create_dropdown_with_default_selection(
        label="Inference run ID",
        key="inference_run_id_single_visualization",
        options=get_inference_run_ids(),
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

    # Create a dictionary with the variable names and associated dimensions to choose from if they
    # have more than one.
    variable_dimension_dict = {COORDINATION_STATS_VARIABLE: [], LATENT_PARAMETERS_VARIABLE: []}
    variable_dimension_dict.update(
        {var: dims for var, dims in model_variables["latent"]}
    )
    variable_dimension_dict.update(
        {var: dims for var, dims in model_variables["observed"]}
    )

    # Prefix + (variable name, mode)
    variable_names = [DropDownOptionWithInferenceMode(v[0], "posterior", "[LATENT]") for v in
                      model_variables["latent"]]
    variable_names += [DropDownOptionWithInferenceMode(v[0], "posterior", "[OBSERVED]") for v in
                       model_variables["observed"]]
    variable_names += [
        DropDownOptionWithInferenceMode(v[0], "posterior_predictive", "[POSTERIOR_PREDICTIVE]") for
        v in
        model_variables["posterior_predictive"]]
    variable_names += [
        DropDownOptionWithInferenceMode(COORDINATION_STATS_VARIABLE, "posterior", "[EXTRA]"),
        DropDownOptionWithInferenceMode(LATENT_PARAMETERS_VARIABLE, "posterior", "[EXTRA]")]
    variable_names.sort(key=lambda x: x.name)
    tab1, tab2 = st.columns(2)
    with tab1:
        st.write("## Model variable")
        model_variable_left = create_dropdown_with_default_selection(
            label="Variable",
            key="model_variable_visualization_left",
            options=variable_names
        )

        model_variable_dimension_left = 0
        if model_variable_left and len(variable_dimension_dict[model_variable_left.name]) > 1:
            model_variable_dimension_left = st.selectbox(
                "Dimension",
                key="model_variable_visualization_left_dimension",
                options=variable_dimension_dict[model_variable_left.name]
            )

    with tab2:
        st.write("## Model variable")
        model_variable_right = create_dropdown_with_default_selection(
            label="Variable",
            key="model_variable_visualization_right",
            options=variable_names
        )

        model_variable_dimension_right = 0
        if model_variable_right and len(variable_dimension_dict[model_variable_right.name]) > 1:
            model_variable_dimension_right = st.selectbox(
                "Dimension",
                key="model_variable_visualization_right_dimension",
                options=variable_dimension_dict[model_variable_right.name]
            )

    st.divider()

    if execution_params_dict:
        for experiment_id in selected_experiment_ids:
            tab1, tab2 = st.columns(2)
            with tab1:
                if model_variable_left:
                    _populate_variable_pane(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        variable_name=model_variable_left.name,
                        variable_dimension=model_variable_dimension_left,
                        mode=model_variable_left.inference_mode
                    )
            with tab2:
                if model_variable_right:
                    _populate_variable_pane(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        variable_name=model_variable_right.name,
                        variable_dimension=model_variable_dimension_right,
                        mode=model_variable_right.inference_mode
                    )


def _populate_variable_pane(run_id: str,
                            experiment_id: str,
                            variable_name: str,
                            variable_dimension: str,
                            mode: str):
    """
    Populates pane with a list of inference plots for a model variable/parameters, or general
    coordination statistics.

    @param run_id: ID of the inference run.
    @param experiment_id: experiment ID in the run to analyze.
    @param variable_name: name of the variable to plot.
    @param variable_dimension: the dimension to plot if there's more than one.
    @param mode: inference mode: posterior, prior_check or posterior_predictive
    """
    inference_dir = st.session_state["inference_results_dir"]
    st.write(f"### {experiment_id}")

    experiment_dir = f"{inference_dir}/{run_id}/{experiment_id}"
    idata = InferenceData.from_trace_file_in_directory(experiment_dir)

    if not idata:
        st.write(":red[No inference data found.]")
        return

    if variable_name == COORDINATION_STATS_VARIABLE:
        means = idata.average_posterior_samples("coordination", return_std=False)
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
        fig = plot_curve(
            variable_name=variable_name,
            dimension=variable_dimension,
            inference_data=idata,
            mode=mode
        )
        # margin_settings = dict(
        #     l=0,  # Adjust the left margin
        #     r=0,  # Adjust the right margin
        #     b=50,  # Adjust the bottom margin
        #     t=0,  # Adjust the top margin
        # ),
        # fig.update_layout(margin=margin_settings)
