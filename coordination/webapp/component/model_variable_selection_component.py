import uuid

import streamlit as st
from coordination.webapp.widget.drop_down import DropDownOption, DropDown
from coordination.webapp.entity.inference_run import InferenceRun
from coordination.webapp.entity.model_variable import ModelVariableInfo


class ModelVariableSelectionComponent:
    """
    Represents a component that displays a collection of model variables collected from the
    inference data of some experiment in an inference run. If a variable has more than one
    dimension and is  not a parameter variable, named dimensions are presented in under the
    variable selector for further filtering.
    """

    def __init__(self, inference_run: InferenceRun):
        """
        Creates the component.

        @param inference_run: object containing info about an inference run.
        """
        self.inference_run = inference_run

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_model_variable_: ModelVariableInfo = None
        self.execution_params_dict_ = None

    def create_component(self):
        """
        Creates drop downs for model variables evaluated in the inference run and associated
        dimensions.
        """
        st.write("## Model variable")
        options = [
            ModelVariableDropDownOption(f"{group.upper}", var_info) for group, var_info in
            self.inference_run.model_variables.items()
        ]
        self.selected_model_variable_ = DropDown(
            label="Variable",
            options=options
        ).model_variable_info

        create_dropdown_with_default_selection(
            label="Variable",
            key=f"model_variable_selector_{key_suffix}",
            options=_get_drop_down_model_variable_options(run_id)
        )

    dimension_name = None
    if selected_variable and selected_variable.dimension_names and len(
            selected_variable.dimension_names) > 1:
        # Selector for dimension if the variable has multiple dimensions.
        dimension_name = st.selectbox(
            "Dimension",
            key=f"dimension_name_selector_{key_suffix}",
            options=selected_variable.dimension_names
        )

    self.selected_run_id_ = DropDown(
        label="Inference run ID",
        options=self._get_inference_run_ids()
    ).create()

    if self.selected_run_id_:
        # Display the execution params for the inference run
        self.execution_params_dict_ = get_execution_params(self.selected_run_id_)
        if self.execution_params_dict_:
            st.json(self.execution_params_dict_, expanded=False)


def _get_inference_run_ids() -> List[str]:
    """
    Gets a list of inference run IDs from the list of directories under an inference folder.

    @return: list of inference run ids.
    """
    if os.path.exists(self.inference_dir):
        run_ids = [run_id for run_id in os.listdir(inference_dir) if
                   os.path.isdir(f"{self.inference_dir}/{run_id}")]

        # Display on the screen from the most recent to the oldest.
        return sorted(run_ids, reverse=True)

    return []


class ModelVariableDropDownOption(DropDownOption):
    """
    This class represents a dropdown option for a model variable. It contains an extra parameter
    to store extra info about a variable to be retrieved when one variable is selected from the
    dropdown.
    """

    def __init__(self,
                 prefix: str,
                 model_variable_info: ModelVariableInfo):
        """
        Creates a dropdown option.

        @param prefix: a prefix to add before each variable name for better identification.
        @param model_variable_info: a model variable object with extra information about a
            variable.
        """
        super().__init__(model_variable_info.name, prefix)
        self.model_variable_info = model_variable_info
