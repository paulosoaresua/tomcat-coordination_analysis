from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_series(x: np.ndarray,
                y: np.ndarray,
                y_std: np.ndarray = None,
                label: str = "",
                include_bands: bool = False,
                value_bounds: Optional[Tuple[float, float]] = None,
                ax: Optional[plt.axis] = None,
                **kwargs):
    # colormap = ListedColormap(SPRING_PASTELS_COLOR_PALETTE)
    ax.plot(x,
            y,
            markerfacecolor=kwargs.pop("markerfacecolor", "None"),
            marker=kwargs.pop("marker", "o"),
            # cmap=kwargs.pop("cmap", colormap),
            label=label,
            **kwargs)
    if include_bands:
        lower_band = y - y_std
        if value_bounds is not None and value_bounds[0] is not None:
            lower_band = np.maximum(lower_band, value_bounds[0])

        upper_band = np.minimum(y + y_std, 1)
        if value_bounds is not None and value_bounds[1] is not None:
            upper_band = np.maximum(upper_band, value_bounds[1])

        ax.fill_between(x, lower_band, upper_band, alpha=0.5)  # , cmap=colormap)
