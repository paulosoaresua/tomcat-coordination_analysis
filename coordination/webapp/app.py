import os

import streamlit as st

from coordination.common.constants import DEFAULT_INFERENCE_RESULTS_DIR
from coordination.webapp.pages.header import create_header
from coordination.webapp.pages.run_page import create_run_page
from coordination.webapp.pages.visualization_page import \
    create_visualization_per_run_page

st.set_page_config(
    page_title="Coordination Processes",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("Coordination Processes")

if "inference_results_dir" not in st.session_state:
    st.session_state["inference_results_dir"] = os.getenv(
        "INFERENCE_RESULTS_DIR", DEFAULT_INFERENCE_RESULTS_DIR
    )

# Progress bars change colors according to the percentage of the progress.
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #fc8787, #00ffa2);
        }
    </style>""",
    unsafe_allow_html=True,
)

create_header()

tab1, tab2, tab3 = st.tabs(["Visualization per Run", "Run vs Run", "Run"])

with tab1:
    create_visualization_per_run_page()

with tab2:
    st.write("Not implemented yet")
    # create_run_vs_run_page()

with tab3:
    create_run_page()
