from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from xarray import DataArray


def plot_series(
    x: Union[np.ndarray, DataArray],
    y: Union[np.ndarray, DataArray],
    y_std: Union[np.ndarray, DataArray] = None,
    label: str = "",
    include_bands: bool = False,
    value_bounds: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.axis] = None,
    **kwargs
) -> plt.axis:
    """
    Plots a time series with optional error bands.

    @param x: time steps.
    @param y: values for each time step.
    @param y_std: standard deviation of the values for error band plotting.
    @param label: label of the series.
    @param include_bands: whether to include bands or not.
    @param value_bounds: optional tuple containing min and max values to be plotted. Values out of
        the range will be capped.
    @param ax: an optional axis to plot onto. If not provided, it will be created.
    @param kwargs: optional extra parameters to be passed to the matplotlib.pyplot.plt function.
    @return: plot axis.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if include_bands:
        lower_band = y - y_std
        if value_bounds is not None and value_bounds[0] is not None:
            lower_band = np.maximum(lower_band, value_bounds[0])

        upper_band = y + y_std
        if value_bounds is not None and value_bounds[1] is not None:
            upper_band = np.minimum(upper_band, value_bounds[1])

        ax.fill_between(
            x, lower_band, upper_band, color=kwargs.get("color", None), alpha=0.5
        )

    if (
        value_bounds is not None
        and value_bounds[0] is not None
        and value_bounds[1] is not None
    ):
        ax.set_ylim(value_bounds)

    ax.plot(
        x,
        y,
        markerfacecolor=kwargs.pop("markerfacecolor", "None"),
        marker=kwargs.pop("marker", "o"),
        label=label,
        **kwargs
    )

    return ax
