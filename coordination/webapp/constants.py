import os

import plotly.express as px

# Holds temporary files related to the webapp execution
APP_RUN_DIR = os.getenv("APP_RUN_DIR", ".webapp")
EVALUATIONS_DIR = os.getenv("EVAL_DIR", "evaluations")
INFERENCE_PARAMETERS_DIR = f"{APP_RUN_DIR}/inference/execution_params"
INFERENCE_TMP_DIR = f"{APP_RUN_DIR}/inference/tmp"
DEFAULT_COLOR_PALETTE = px.colors.qualitative.Pastel1
DEFAULT_PLOT_MARGINS = dict(l=0, r=0, t=30, b=10)
INFERENCE_RESULTS_DIR_STATE_KEY = "inference_results_dir"
AVAILABLE_EXPERIMENTS_STATE_KEY = "available_experiments"
NUM_LAST_RUNS = 5
