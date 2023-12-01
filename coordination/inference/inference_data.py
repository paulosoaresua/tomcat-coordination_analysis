from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from coordination.common.plot import plot_series

TRACE_FILENAME = "inference_data.pkl"


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

    @property
    def posterior_latent_variables(self) -> List[str]:
        """
        Gets names of latent variables that were sampled during model fit.

        We consider latent variables those that are in the list of latent variables and that have
        a time dimension attached to them. The remaining latent variables are considered latent
        parameters and can be retrieved with the method posterior_latent_parameters.

        @return: list of latent variable names
        """
        return self._get_vars(mode="posterior", with_time_dimension=True)

    @property
    def posterior_predictive_variables(self) -> List[str]:
        """
        Gets names of variables that were sampled during posterior predictive check.

        We consider variables those that are in the list of posterior predictive variables and
        that have a time dimension attached to them. The remaining latent variables are considered
        latent parameters and can be retrieved with the method posterior_predictive_parameters.

        @return: list of posterior predictive variable names
        """
        return self._get_vars(mode="posterior_predictive", with_time_dimension=True)

    @property
    def observed_variables(self) -> List[str]:
        """
        Gets names of variables that were observed during model fit.

         We consider observed variables those that are in the list of observed variables and that
         have a time dimension attached to them. The remaining observed variables are considered
         parameters parameters and can be retrieved from the model config bundle in the execution
         params of an inference run.

        @return: list of latent variable names
        """
        return self._get_vars(mode="observed_data", with_time_dimension=True)

    def posterior_latent_parameters(self) -> List[str]:
        """
        Gets names of latent parameter variables that were inferred during model fit.

        We consider parameter variables those that do not have a time dimension attached to them.

        @return: list of latent parameter variable names.
        """
        return self._get_vars(mode="posterior", with_time_dimension=False)

    def _get_vars(self, mode: str, with_time_dimension: bool) -> List[str]:
        """
        Gets a list of variable names from a specific mode with or without an associated time
        dimension.

        @param mode: @param mode: one of the following modes in the trace "posterior",
            "prior_check", "posterior_predictive", "observed_data".
        @param with_time_dimension: whether the variable has a time dimension associated to it.
        @return: list of variable names.
        """
        if mode not in self.trace:
            return []

        var_names = []
        for var_name in self.trace[mode].data_vars:
            var = self.trace[mode].data_vars[var_name]
            if with_time_dimension:
                if len([dim for dim in var.dims if "time" in dim]) > 0:
                    var_names.append(var_name)
            else:
                if len([dim for dim in var.dims if "time" in dim]) == 0:
                    var_names.append(var_name)

        return var_names

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

    def plot_parameter_posterior(self) -> Optional[plt.Figure]:
        """
        Plot posteriors of the latent parameters in the model.
        """

        var_names = self.posterior_latent_parameters()
        if len(var_names) > 0:
            var_names = sorted(var_names)
            axes = az.plot_trace(self.trace, var_names=var_names)
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            return fig

        return None

    def average_posterior_samples(
            self, variable_uuid: str, return_std: bool
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples of a variable's posterior distribution.

        @param variable_uuid: unique identifier of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation per time step.
        """

        return self._average_samples(variable_uuid, return_std, "posterior")

    def average_posterior_predictive_samples(
            self, variable_uuid: str, return_std: bool
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples of a variable's posterior predictive distribution.

        @param variable_uuid: unique identifier of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation per time step.
        """

        return self._average_samples(variable_uuid, return_std, "posterior_predictive")

    def _average_samples(
            self, variable_uuid: str, return_std: bool, mode: str
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples in the trace.

        @param variable_uuid: unique identifier of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @param mode: one of the following modes in the trace "posterior", "prior_check",
            "posterior_predictive".
        @raise ValueError: if the mode is not in the inference data.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation per time step.
        """
        if mode not in self.trace:
            raise ValueError(f"{mode} samples not found in the inference data.")

        if variable_uuid in self.trace[mode]:
            samples = self.trace[mode][variable_uuid]

            mean_values = samples.mean(dim=["chain", "draw"])

            if return_std:
                std_values = samples.std(dim=["chain", "draw"])
                return mean_values, std_values

            return mean_values
        elif variable_uuid in self.trace.observed_data:
            samples = self.trace.observed_data[variable_uuid]

            mean_values = samples

            if return_std:
                return mean_values, None

            return mean_values
        else:
            raise ValueError(
                f"Variable ({variable_uuid}) not found in the inference data (mode = {mode}).")

    def plot_time_series_posterior(
            self,
            variable_uuid: str,
            include_bands: bool,
            value_bounds: Optional[Tuple[float, float]] = None,
            ax: Optional[plt.axis] = None,
            dimension: Union[int, str] = 0,
            **kwargs,
    ) -> plt.axis:
        """
        Plots the time series of samples draw from the posterior distribution.

        @param variable_uuid: variable to plot.
        @param include_bands: whether to include error bands.
        @param value_bounds: minimum and maximum values to limit values to a range.
        @param ax: axis to plot on. It will be created if not provided.
        @param dimension: index or name of the dimension axis to plot.
        @param kwargs: extra parameters to pass to the plot function.
        @return: plot axis.
        """

        if ax is None:
            plt.figure()
            ax = plt.gca()

        means, stds = self.average_posterior_samples(
            variable_uuid=variable_uuid, return_std=True
        )

        time_steps = np.arange(means.shape[-1])
        if len(means.shape) == 1:
            # Coordination plot
            plot_series(
                x=time_steps,
                y=means,
                y_std=stds,
                label=None,
                include_bands=include_bands,
                value_bounds=value_bounds,
                ax=ax,
                **kwargs,
            )
            ax.set_ylabel("Coordination")
        elif len(means.shape) == 2:
            # Serial variable
            subject_indices = np.array(
                [
                    int(x.split("#")[0])
                    for x in getattr(means, f"{variable_uuid}_time").data
                ]
            )
            time_steps = np.array(
                [
                    int(x.split("#")[1])
                    for x in getattr(means, f"{variable_uuid}_time").data
                ]
            )
            subjects = sorted(list(set(subject_indices)))
            for s in subjects:
                idx = [i for i, subject in enumerate(subject_indices) if subject == s]
                y = means.loc[dimension][idx] if isinstance(dimension, str) else means[
                    dimension, idx]
                y_std = stds.loc[dimension][idx] if isinstance(dimension, str) else stds[
                    dimension, idx]
                plot_series(
                    x=time_steps[idx],
                    y=y,
                    y_std=y_std,
                    label=f"Subject {s}",
                    include_bands=include_bands,
                    value_bounds=value_bounds,
                    ax=ax,
                    **kwargs,
                )
            ax.set_ylabel(
                getattr(means, f"{variable_uuid}_dimension").data[dimension_idx]
            )
        else:
            # Non-serial variable
            for s in range(means.shape[0]):
                y = means[s].loc[dimension] if isinstance(dimension, str) else means[s, dimension]
                y_std = stds[s].loc[dimension] if isinstance(dimension, str) else stds[
                    s, dimension]
                plot_series(
                    x=time_steps,
                    y=y,
                    y_std=y_std,
                    label=f"Subject {s}",
                    include_bands=include_bands,
                    value_bounds=value_bounds,
                    ax=ax,
                    **kwargs,
                )
            ax.set_ylabel(
                getattr(means, f"{variable_uuid}_dimension").data[dimension_idx]
            )

        ax.set_xlabel("Time Step")
        ax.spines[["right", "top"]].set_visible(False)

        return ax

    def save_to_directory(self, directory: str):
        """
        Save inference data. We save the trace since that's more stable due to be a third-party
        object. In other words, if we change the inference data class we don't lose the save data
        because of incompatibility.

        @param directory: directory where the trace file must be saved.
        """
        with open(f"{directory}/{TRACE_FILENAME}", "wb") as f:
            pickle.dump(self.trace, f)

    @classmethod
    def from_trace_file_in_directory(cls, directory: str) -> Optional[InferenceData]:
        filepath = f"{directory}/{TRACE_FILENAME}"
        if not os.path.exists(directory):
            return None

        try:
            with open(filepath, "rb") as f:
                return cls(pickle.load(f))
        except Exception:
            return None

    def get_dimension_coordinates(self, variable_name: str) -> List[str]:
        """
        Gets the dimension coordinate values for a given variable.

        @param variable_name: variable names.
        @return: list of coordinates of the dimension axis of a variable.
        """
        dim_coordinate = f"{variable_name}_dimension"

        if dim_coordinate in self.trace.observed_data:
            return self.trace.observed_data[dim_coordinate].data.tolist()
        elif "posterior" in self.trace and dim_coordinate in self.trace.posterior:
            return self.trace.posterior[dim_coordinate].data.tolist()
        else:
            return []

    def get_log_probs(self) -> np.ndarray:
        """
        Gets a matrix containing estimated log-probabilities of the model for each chain (rows)
        and draw (columns).

        @return: a matrix of log-probabilities.
        """
        self.trace.sample_stats.lp.to_numpy()
