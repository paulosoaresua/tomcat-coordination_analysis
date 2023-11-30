import streamlit as st


def create_header():
    inference_results = st.text_input(label="Inference Results Directory",
                                      value=st.session_state["inference_results_dir"])
    submit = st.button(label="Update Directory")
    if submit:
        st.session_state["inference_results_dir"] = inference_results
