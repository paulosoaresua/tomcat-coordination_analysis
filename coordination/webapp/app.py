import os

import streamlit as st

from coordination.common.constants import DEFAULT_INFERENCE_RESULTS_DIR
from coordination.webapp.component.header import Header
from coordination.webapp.pages.new_run import NewRun
from coordination.webapp.pages.progress import Progress
from coordination.webapp.pages.run_vs_run import RunVsRun
from coordination.webapp.pages.single_run import SingleRun

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

Header().create_component()

tab1, tab2, tab3, tab4 = st.tabs(["Single Run", "Run vs Run", "New Run", "Progress"])

with tab1:
    SingleRun(page_key="single_run_tab").create_page()

with tab2:
    RunVsRun(page_key="run_vs_run_tab").create_page()

with tab3:
    NewRun(page_key="new_run_tab").create_page()

with tab4:
    Progress(page_key="progress_tab").create_page()
