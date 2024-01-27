import os

import pandas as pd
import streamlit as st

from coordination.webapp.constants import EVALUATIONS_DIR
from coordination.inference.inference_run import InferenceRun


class EvaluationResults:
    """
    Represents a component that displays evaluation results for an inference run.
    """

    def __init__(self, component_key: str, inference_run: InferenceRun):
        """
        Creates the component.

        @param component_key: unique identifier for the component in a page.
        @param inference_run: object containing info about an inference run.
        """
        self.component_key = component_key
        self.inference_run = inference_run

    def create_component(self):
        """
        Displays evaluation results in different forms depending on the variable selected.
        """
        data = {"images": [], "tables": []}
        for filename in os.listdir(f"{EVALUATIONS_DIR}/{self.inference_run.run_id}"):
            if filename[-3:] == "png":
                data["images"].append(filename)
            elif filename[-3:] == "csv":
                data["tables"].append(filename)

        st.header("Tables", divider="gray")
        for filename in sorted(data["tables"]):
            filepath = f"{EVALUATIONS_DIR}/{self.inference_run.run_id}/{filename}"
            st.write(f"**{filename}**")
            st.write(pd.read_csv(filepath))

        st.header("Images", divider="gray")
        for filename in sorted(data["images"]):
            filepath = f"{EVALUATIONS_DIR}/{self.inference_run.run_id}/{filename}"
            st.write(f"**{filename}**")
            st.image(filepath)
