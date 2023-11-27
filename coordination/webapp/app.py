import os

import streamlit as st

from coordination.webapp.pages.run_page import create_run_page
from coordination.webapp.pages.visualization_page import create_visualization_page

from coordination.common.constants import DEFAULT_INFERENCE_RESULTS_DIR

st.set_page_config(page_title="Coordination Processes",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="collapsed",
                   menu_items=None)

st.title("Coordination Processes")

if "inference_results_dir" not in st.session_state:
    st.session_state["inference_results_dir"] = os.getenv("INFERENCE_RESULTS_DIR",
                                                          DEFAULT_INFERENCE_RESULTS_DIR)

# create_header()

tab1, tab2 = st.tabs(["Run", "Visualization"])

with tab1:
    create_run_page()

with tab2:
    create_visualization_page()
