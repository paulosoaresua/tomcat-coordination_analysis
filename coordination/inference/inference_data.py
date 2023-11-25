from __future__ import annotations

import pickle
from typing import Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coordination.common.plot import plot_series


class InferenceData:
    def __init__(self, trace: az.InferenceData):
        self.trace = trace

    @property
    def num_divergences(self) -> int:
        """
        Gets the number of divergences in a trace.

        @return: number of divergences.
        """
        return int(self.trace.sample_stats.diverging.sum(dim=["chain", "draw"]))

    @property
    def num_posterior_samples(self) -> int:
        """
        Gets total number of samples in the posterior inference.

        @return: total number of samples.
        """
        num_chains = self.trace["posterior"].sizes["chain"]
        num_samples_per_chain = self.trace["posterior"].sizes["draw"]

        return num_chains * num_samples_per_chain

    def generate_convergence_summary(self) -> pd.DataFrame:
        """
        Estimates Rhat distribution for the latent variables in the posterior inference data.
        @return: Rhat distribution per latent variable.
        """

        header = ["variable", "mean_rhat", "std_rhat"]

        rhat = az.rhat(self.trace)
        data = []
        for var, values in rhat.data_vars.items():
            entry = [var, values.to_numpy().mean(), values.to_numpy().std()]
            data.append(entry)

        return pd.DataFrame(data, columns=header)

    def add(self, inference_data: InferenceData):
        """
        Adds another inference data.

        @param inference_data: inference data.
        """
        self.trace.extend(inference_data.trace)

    def plot_parameter_posterior(self):
        """
        Plot posteriors of the latent parameters in the model.
        """

        # Get from the list of variables in the posterior trace that do not have a time dimension
        # attached to them.
        var_names = []
        for var_name in self.trace["posterior"].data_vars:
            var = self.trace["posterior"].data_vars[var_name]
            if len([dim for dim in var.dims if "time" in dim]) == 0:
                var_names.append(var_name)

        if len(var_names) > 0:
            var_names = sorted(var_names)
            axes = az.plot_trace(self.trace, var_names=var_names)
            fig = axes.ravel()[0].figure
            fig.tight_layout()

    def average_samples(
        self, variable_uuid: str, return_std: bool
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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

    def plot_time_series_posterior(
        self,
        variable_uuid: str,
        include_bands: bool,
        value_bounds: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.axis] = None,
        dimension_idx: int = 0,
        **kwargs,
    ) -> plt.axis:
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

        mean_values, std_values = self.average_samples(
            variable_uuid=variable_uuid, return_std=True
        )

        time_steps = np.arange(mean_values.shape[-1])
        if len(mean_values.shape) == 1:
            # Coordination plot
            plot_series(
                x=time_steps,
                y=mean_values,
                y_std=std_values,
                label=None,
                include_bands=include_bands,
                value_bounds=value_bounds,
                ax=ax,
                **kwargs,
            )
            ax.set_ylabel("Coordination")
        elif len(mean_values.shape) == 2:
            # Serial variable
            subject_indices = np.array(
                [
                    int(x.split("#")[0])
                    for x in getattr(mean_values, f"{variable_uuid}_time").data
                ]
            )
            time_steps = np.array(
                [
                    int(x.split("#")[1])
                    for x in getattr(mean_values, f"{variable_uuid}_time").data
                ]
            )
            subjects = sorted(list(set(subject_indices)))
            for s in subjects:
                idx = [i for i, subject in enumerate(subject_indices) if subject == s]
                plot_series(
                    x=time_steps[idx],
                    y=mean_values[dimension_idx, idx],
                    y_std=std_values[dimension_idx, idx],
                    label=f"Subject {s}",
                    include_bands=include_bands,
                    value_bounds=value_bounds,
                    ax=ax,
                    **kwargs,
                )
            ax.set_ylabel(
                getattr(mean_values, f"{variable_uuid}_dimension").data[dimension_idx]
            )
        else:
            # Non-serial variable
            for s in range(mean_values.shape[0]):
                plot_series(
                    x=time_steps,
                    y=mean_values[s, dimension_idx],
                    y_std=std_values[s, dimension_idx],
                    label=f"Subject {s}",
                    include_bands=include_bands,
                    value_bounds=value_bounds,
                    ax=ax,
                    **kwargs,
                )
            ax.set_ylabel(
                getattr(mean_values, f"{variable_uuid}_dimension").data[dimension_idx]
            )

        ax.set_xlabel("Time Step")
        ax.spines[["right", "top"]].set_visible(False)

        return ax

    def save(self, filepath: str):
        """
        Save inference data. We save the trace since that's more stable due to be a third-party
        object. In other words, if we change the inference data class we don't lose the save data
        because of incompatibility.

        @param filepath: path of the file.
        """
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(self.trace, f)
