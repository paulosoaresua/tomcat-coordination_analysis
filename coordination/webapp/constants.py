from coordination.common.constants import DEFAULT_INFERENCE_RESULTS_DIR
import plotly.express as px

REFRESH_RATE = 10  # in seconds

RUN_DIR = ".run"
INFERENCE_PARAMETERS_DIR = f"{DEFAULT_INFERENCE_RESULTS_DIR}/inference_params"
DEFAULT_COLOR_PALETTE = px.colors.qualitative.Pastel1
DEFAULT_PLOT_MARGINS = dict(l=0, r=0, t=30, b=10)
