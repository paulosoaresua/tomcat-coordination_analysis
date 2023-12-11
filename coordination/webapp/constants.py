import os

import plotly.express as px

REFRESH_RATE = 10  # in seconds

# Holds temporary files related to the webapp execution
APP_RUN_DIR = os.getenv("APP_RUN_DIR", ".webapp")
INFERENCE_PARAMETERS_DIR = f"{APP_RUN_DIR}/inference/execution_params"
INFERENCE_TMP_DIR = f"{APP_RUN_DIR}/inference/tmp"
DEFAULT_COLOR_PALETTE = px.colors.qualitative.Pastel1
DEFAULT_PLOT_MARGINS = dict(l=0, r=0, t=30, b=10)
INFERENCE_RESULTS_DIR_STATE_KEY = "inference_results_dir"
AVAILABLE_EXPERIMENTS_STATE_KEY = "available_experiments"
