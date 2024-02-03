import streamlit as st

from coordination.common.config import settings
from coordination.webapp.component.header import Header
from coordination.webapp.constants import (AVAILABLE_EXPERIMENTS_STATE_KEY,
                                           INFERENCE_RESULTS_DIR_STATE_KEY,
                                           WEBAPP_RUN_DIR_STATE_KEY,
                                           EVALUATIONS_DIR_STATE_KEY,
                                           DATA_DIR_STATE_KEY)
from coordination.webapp.pages.new_run import NewRun
from coordination.webapp.pages.progress import Progress
from coordination.webapp.pages.run_vs_run import RunVsRun
from coordination.webapp.pages.run_vs_run_evaluations import \
    RunVsRunEvaluations
from coordination.webapp.pages.single_run import SingleRun
from coordination.webapp.utils import disable_sidebar

st.set_page_config(
    page_title="Coordination Processes",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("Coordination Processes")
disable_sidebar()

if INFERENCE_RESULTS_DIR_STATE_KEY not in st.session_state:
    st.session_state[INFERENCE_RESULTS_DIR_STATE_KEY] = settings.inferences_dir

if EVALUATIONS_DIR_STATE_KEY not in st.session_state:
    st.session_state[EVALUATIONS_DIR_STATE_KEY] = settings.evaluations_dir

if DATA_DIR_STATE_KEY not in st.session_state:
    st.session_state[DATA_DIR_STATE_KEY] = settings.data_dir

if WEBAPP_RUN_DIR_STATE_KEY not in st.session_state:
    st.session_state[WEBAPP_RUN_DIR_STATE_KEY] = settings.webapp_run_dir

if AVAILABLE_EXPERIMENTS_STATE_KEY not in st.session_state:
    st.session_state[AVAILABLE_EXPERIMENTS_STATE_KEY] = []

Header.create_component()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Single Run", "Run vs Run", "Evaluations", "New Run", "Progress"]
)

with tab1:
    SingleRun(page_key="single_run_tab").create_page()

with tab2:
    RunVsRun(page_key="run_vs_run_tab").create_page()

with tab3:
    RunVsRunEvaluations(page_key="evaluations_tab").create_page()

with tab4:
    NewRun(page_key="new_run_tab").create_page()

with tab5:
    Progress(page_key="progress_tab").create_page()
