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

    def __init__(self, component_key: str, inference_run: InferenceRun):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_run: object containing info about an inference run.
        """
        self.component_key = component_key
        self.inference_run = inference_run

        # Values saved within page loading and available to the next components to be loaded.
        # Not persisted through the session.
        self.selected_model_variable_: ModelVariableInfo = None
        self.selected_dimension_name_ = None

    def create_component(self):
        """
        Creates drop downs for model variables evaluated in the inference run and associated
        dimensions.
        """
        st.write("## Model variable")
        options = []
        for group, variables in self.inference_run.model_variables.items():
            for var_info in variables:
                options.append(ModelVariableDropDownOption(prefix=f"[{group.upper()}]",
                                                           model_variable_info=var_info))

        # Add extra options for inference stats and parameter trace plot image
        options.append(ModelVariableDropDownOption(
            prefix="[EXTRA]",
            model_variable_info=ModelVariableInfo(
                name="Inference Stats",
                inference_mode="inference_stats",
                dimension_names=[]
            )))
        options.append(ModelVariableDropDownOption(
            prefix="[EXTRA]",
            model_variable_info=ModelVariableInfo(
                name="Parameter Trace Plot",
                inference_mode="parameter_trace",
                dimension_names=[]
            )))
        options.sort(key=lambda x: (x.prefix, x.name))

        selected_option = DropDown(
            label="Variable",
            key=f"{self.component_key}_model_variable_dropdown",
            options=options
        ).create()

        self.selected_model_variable_ = selected_option.model_variable_info if \
            selected_option else None

        if self.selected_model_variable_:
            self.selected_dimension_name_ = None
            if self.selected_model_variable_ and \
                    self.selected_model_variable_.num_named_dimensions > 1:
                # Selector for dimension if the variable has multiple dimensions.
                self.selected_dimension_name_ = st.selectbox(
                    "Dimension",
                    key=f"{self.component_key}_model_variable_dimension_dropdown",
                    options=self.selected_model_variable_.dimension_names
                )


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
        Creates a dropdown option for model variable selection.

        @param prefix: a prefix to add before each variable name for better identification.
        @param model_variable_info: a model variable object with extra information about a
            variable.
        """
        super().__init__(model_variable_info.name, prefix)
        self.model_variable_info = model_variable_info
