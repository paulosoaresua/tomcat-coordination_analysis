import os

import plotly.express as px

# Session keys
INFERENCE_RESULTS_DIR_STATE_KEY = "inference_results_dir"
EVALUATIONS_DIR_STATE_KEY = "evaluations_dir"
DATA_DIR_STATE_KEY = "data_dir"
WEBAPP_RUN_DIR_STATE_KEY = "webapp_run_dir"
AVAILABLE_EXPERIMENTS_STATE_KEY = "available_experiments"

# Holds temporary files related to the webapp execution
DEFAULT_COLOR_PALETTE = px.colors.qualitative.Pastel1
DEFAULT_PLOT_MARGINS = dict(l=0, r=0, t=30, b=10)
NUM_LAST_RUNS = 5
REFRESH_RATE = 10  # in seconds
