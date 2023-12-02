import streamlit as st

class RunVsRun:

    def create_page(self):
        """
        Creates a run x run page for comparison of experiments across two different inference runs.
        """
        tab_left, tab_right = st.columns(2)
        with tab_left:
            self._populate_column()

        with tab_right:
            self._populate_column()

    def _populate_column(self):

