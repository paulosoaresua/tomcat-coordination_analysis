from typing import Optional, Tuple, Union

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from coordination.common.style import SPRING_PASTELS_COLOR_PALETTE
from coordination.common.plot import plot_series


class InferenceData:

    def __init__(self, trace: az.InferenceData):
        self.trace = trace

    def average_samples(self,
                        variable_uuid: str,
                        return_std: bool) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Gets the mean values from the samples of a variable's posterior distribution.

        @param variable_uuid: unique identifier of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @return: variable's posterior mean and optionally standard deviation per time step.
        """

        samples = self.trace.posterior[variable_uuid]
        mean_values = samples.mean(dim=["chain", "draw"])

        if return_std:
            std_values = samples.std(dim=["chain", "draw"])
            return mean_values, std_values

        return mean_values

    def plot_average_samples(self,
                             variable_uuid: str,
                             include_bands: bool,
                             value_bounds: Optional[Tuple[float, float]] = None,
                             ax: Optional[plt.axis] = None,
                             dimension_idx: int = 0,
                             **kwargs) -> plt.axis:
        """
        Plots the time series of samples draw from the posterior distribution.

        @param variable_uuid: variable to plot.
        @param include_bands: whether to include error bands.
        @param value_bounds: minimum and maximum values to limit values to a range.
        @param ax: axis to plot on. It will be created if not provided.
        @param dimension_idx: index of the dimension axis to plot.
        @param kwargs: extra parameters to pass to the plot function.
        @return: plot axis.
        """

        if ax is None:
            plt.figure()
            ax = plt.gca()

        mean_values, std_values = self.average_samples(variable_uuid=variable_uuid,
                                                       return_std=True)

        dim_names = mean_values.coords.dims
        time_steps = np.arange(mean_values.shape[-1])
        if len(mean_values.shape) == 1:
            # Coordination plot
            plot_series(x=time_steps,
                        y=mean_values,
                        y_std=std_values,
                        label=None,
                        include_bands=include_bands,
                        value_bounds=value_bounds,
                        ax=ax,
                        **kwargs)
            ax.set_ylabel("Coordination")
        elif len(mean_values.shape) == 2:
            # Serial variable
            subject_indices = np.array([int(x.split("#")[0]) for x in
                                        getattr(mean_values, f"{variable_uuid}_time").data])
            time_steps = np.array([int(x.split("#")[1]) for x in
                                   getattr(mean_values, f"{variable_uuid}_time").data])
            subjects = sorted(list(set(subject_indices)))
            for s in subjects:
                idx = [i for i, subject in enumerate(subject_indices) if subject == s]
                plot_series(x=time_steps[idx],
                            y=mean_values[dimension_idx, idx],
                            y_std=std_values[dimension_idx, idx],
                            label=f"Subject {s}",
                            include_bands=include_bands,
                            value_bounds=value_bounds,
                            ax=ax,
                            **kwargs)
            ax.set_ylabel(getattr(mean_values, f"{variable_uuid}_dimension").data[dimension_axis])
        else:
            # Non-serial variable
            for s in range(mean_values.shape[0]):
                plot_series(x=time_steps,
                            y=mean_values[s, dimension_idx],
                            y_std=std_values[s, dimension_idx],
                            label=f"Subject {s}",
                            include_bands=include_bands,
                            value_bounds=value_bounds,
                            ax=ax,
                            **kwargs)
            ax.set_ylabel(getattr(mean_values, f"{variable_uuid}_dimension").data[dimension_axis])

        ax.set_xlabel("Time Step")
        ax.spines[['right', 'top']].set_visible(False)

        return ax
