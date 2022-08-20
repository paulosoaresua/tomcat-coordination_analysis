from typing import Any, List, Optional

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_discrete_coordination(ax: Any, coordination: np.ndarray, color: str, title: str, xaxis_label: str,
                               marker: str = "o"):
    ax.plot(range(len(coordination)), coordination, color=color, marker=marker, linestyle="--")
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel(xaxis_label)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Coordination", "Coordination"])


def add_discrete_coordination_bar(main_ax: Any, coordination_series: List[np.array],
                                  coordination_colors: List[str], no_coordination_color: str = "white",
                                  labels: Optional[List[str]] = None):
    # For now, we limit the number of coordination series to 2
    assert len(coordination_series) <= 2
    assert labels is None or len(coordination_series) == len(labels)
    assert len(coordination_colors) == len(coordination_series)

    if len(coordination_series) == 0:
        # Nothing to be done
        return

    num_series = len(coordination_series)
    color_bar_values = np.zeros((num_series, len(coordination_series[0])))
    relative_height = f"{5 + (10 * (num_series - 1))}%"

    # 1st color: No coordination (0 <= value < 1)
    # 2nd color: Coordination on series 1 (1 <= value < 2)
    # 3rd color: Coordination in series 2 (2 <= value < 3)
    cmap = colors.ListedColormap([no_coordination_color] + coordination_colors)
    bounds = list(range(num_series + 2))
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for i in range(0, num_series):
        # We change the coordination values in the different series to map it them to different colors
        color_bar_values[i] = np.where(coordination_series[i] == 1, i + 1, 0)

    # Splitting the axis to insert the color bar under it
    divider = make_axes_locatable(main_ax)
    color_bar_ax = divider.append_axes("bottom", size=relative_height, pad=0.1, sharex=main_ax)
    color_bar_ax.imshow(color_bar_values, cmap=cmap, norm=norm, aspect="auto")

    # Labels and ticks
    color_bar_ax.set_xlabel(main_ax.get_xlabel())
    main_ax.axes.xaxis.set_visible(False)

    if labels is None or len(labels) == 0:
        color_bar_ax.axes.yaxis.set_visible(False)
    else:
        bar_height = 0.5
        color_bar_ax.set_ylim([0, num_series * bar_height])
        ticks = [i * bar_height + bar_height / 2 for i in range(num_series)]

        color_bar_ax.set_yticks(ticks)
        color_bar_ax.set_yticklabels(labels)
