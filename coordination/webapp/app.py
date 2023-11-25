import streamlit as st

from coordination.webapp.pages.run_page import create_run_page
from coordination.webapp.pages.visualization_page import create_visualization_page

st.set_page_config(page_title="Coordination Processes",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="collapsed",
                   menu_items=None)

st.title("Coordination Processes")

# create_header()

tab1, tab2 = st.tabs(["Run", "Visualization"])

with tab1:
    create_run_page()

with tab2:
    create_visualization_page()
