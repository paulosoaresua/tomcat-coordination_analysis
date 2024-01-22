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
    def posterior_latent_data_variables(self) -> List[str]:
        """
        Gets names of latent data variables that were sampled during model fit.

        We consider latent data variables those that are in the list of latent variables and that
        have a time dimension attached to them. The remaining latent variables are considered
        latent parameter.

        @return: list of latent variable names
        """
        return self._get_vars(inference_mode="posterior", with_time_dimension=True)

    @property
    def prior_predictive_variables(self) -> List[str]:
        """
        Gets names of data and parameter variables that were sampled during prior predictive check.

        @return: list of posterior predictive variable names
        """
        data_variables = self._get_vars(
            inference_mode="prior_predictive", with_time_dimension=True
        )
        parameter_variables = self._get_vars(
            inference_mode="prior_predictive", with_time_dimension=False
        )
        return data_variables + parameter_variables

    @property
    def posterior_predictive_variables(self) -> List[str]:
        """
        Gets names of variables and parameters that were sampled during posterior predictive check.

        @return: list of posterior predictive variable names
        """
        data_variables = self._get_vars(
            inference_mode="posterior_predictive", with_time_dimension=True
        )
        parameter_variables = self._get_vars(
            inference_mode="posterior_predictive", with_time_dimension=False
        )
        return data_variables + parameter_variables

    @property
    def observed_data_variables(self) -> List[str]:
        """
        Gets names of variables that were observed during model fit.

         We consider observed data variables those that are in the list of observed variables and
         that have a time dimension attached to them. The remaining observed variables are
         considered observed parameter variables.

        @return: list of observed data variable names
        """
        return self._get_vars(inference_mode="observed_data", with_time_dimension=True)

    @property
    def posterior_latent_parameter_variables(self) -> List[str]:
        """
        Gets names of latent parameter variables that were inferred during model fit.

        We consider parameter variables those that do not have a time dimension attached to them.

        @return: list of latent parameter variable names.
        """
        return self._get_vars(inference_mode="posterior", with_time_dimension=False)

    def _get_vars(self, inference_mode: str, with_time_dimension: bool) -> List[str]:
        """
        Gets a list of variable names from a specific mode with or without an associated time
        dimension.

        @param inference_mode: one of the posterior, prior_predictive, or posterior_predictive.
        @param with_time_dimension: whether the variable has a time dimension associated to it.
        @return: list of variable names.
        """
        if inference_mode not in self.trace:
            return []

        var_names = []
        for var_name in self.trace[inference_mode].data_vars:
            if with_time_dimension:
                if not self.is_parameter(inference_mode, var_name):
                    var_names.append(var_name)
            elif self.is_parameter(inference_mode, var_name):
                var_names.append(var_name)

        return var_names

    def is_parameter(self, inference_mode: str, variable_name: str) -> bool:
        """
        Checks is a variable is a parameter or not. We consider a variable as a parameter if it
        does not have a time dimension associated to it.

        @param inference_mode: one of the posterior, prior_predictive, or posterior_predictive.
        @param variable_name: name of the variable.
        @return:
        """
        if inference_mode in self.trace:
            if variable_name in self.trace[inference_mode].data_vars:
                var = self.trace[inference_mode].data_vars[variable_name]
                return len([dim for dim in var.dims if "time" in dim]) == 0
            elif variable_name in self.trace.observed_data.data_vars:
                # Look in the list of observed variables
                var = self.trace.observed_data.data_vars[variable_name]
                return len([dim for dim in var.dims if "time" in dim]) == 0
            else:
                raise ValueError(
                    f"Variable ({variable_name}) not found for inference mode ({inference_mode})."
                )
        else:
            raise ValueError(
                f"Inference mode ({inference_mode}) not found in the trace."
            )

    def generate_convergence_summary(self) -> pd.DataFrame:
        """
        Estimates Rhat distribution for the latent variables in the posterior inference data.
        @return: Rhat distribution per latent variable.
        """

        header = ["variable", "converged", "mean_rhat", "std_rhat"]

        rhat = az.rhat(self.trace)
        data = []
        for var, values in rhat.data_vars.items():
            mean = values.to_numpy().mean()
            std = values.to_numpy().std()
            entry = [var, "yes" if mean < 1.1 else "no", mean, std]
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

        var_names = self.posterior_latent_parameter_variables
        if len(var_names) > 0:
            var_names = sorted(var_names)
            axes = az.plot_trace(self.trace, var_names=var_names)
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            return fig

        return None

    def average_posterior_samples(
            self, variable_name: str, return_std: bool
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples of a variable's posterior distribution.

        @param variable_name: name of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation (per time step if it
            is a data variable).
        """

        return self._average_samples(variable_name, return_std, "posterior")

    def average_prior_predictive_samples(
            self, variable_name: str, return_std: bool
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples of a variable's prior predictive distribution.

        @param variable_name: name of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation (per time step if it
            is a data variable).
        """

        return self._average_samples(variable_name, return_std, "prior_predictive")

    def average_posterior_predictive_samples(
            self, variable_name: str, return_std: bool
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples of a variable's posterior predictive distribution.

        @param variable_name: name of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation (per time step if it
            is a data variable).
        """

        return self._average_samples(variable_name, return_std, "posterior_predictive")

    def _average_samples(
            self, variable_name: str, return_std: bool, inference_mode: str
    ) -> Union[Tuple[xarray.DataArray, xarray.DataArray], xarray.DataArray]:
        """
        Gets the mean values from the samples in the trace.

        @param variable_name: name of the variable with the samples to average.
        @param return_std: whether to return a tuple with the mean values and standard deviation or
            just the mean values.
        @param inference_mode: one of the posterior, prior_predictive, or posterior_predictive.
        @raise ValueError: if the inference mode is not in the inference data.
        @raise ValueError: if the variable is not in the inference data as latent or observed.
        @return: variable's posterior mean and optionally standard deviation per time step.
        """
        if inference_mode not in self.trace:
            raise ValueError(
                f"{inference_mode} samples not found in the inference data."
            )

        if variable_name in self.trace[inference_mode]:
            samples = self.trace[inference_mode][variable_name]

            mean_values = samples.mean(dim=["chain", "draw"])

            if return_std:
                std_values = samples.std(dim=["chain", "draw"])
                return mean_values, std_values

            return mean_values
        elif variable_name in self.trace.observed_data:
            samples = self.trace.observed_data[variable_name]

            mean_values = samples

            if return_std:
                return mean_values, None

            return mean_values
        else:
            raise ValueError(
                f"Variable ({variable_name}) not found in the inference mode ({inference_mode})."
            )

    def plot_time_series_posterior(
            self,
            variable_name: str,
            include_bands: bool,
            value_bounds: Optional[Tuple[float, float]] = None,
            ax: Optional[plt.axis] = None,
            dimension: Union[int, str] = 0,
            show_time_in_coordination_scale: bool = True,
            **kwargs,
    ) -> plt.axis:
        """
        Plots the time series of samples draw from the posterior distribution.

        @param variable_name: name of the variable to plot.
        @param include_bands: whether to include error bands.
        @param value_bounds: minimum and maximum values to limit values to a range.
        @param ax: axis to plot on. It will be created if not provided.
        @param dimension: index or name of the dimension axis to plot.
        @param show_time_in_coordination_scale: whether to display time in coordination scale or
            in the component's scale.
        @param kwargs: extra parameters to pass to the plot function.
        @return: plot axis.
        """

        if ax is None:
            plt.figure()
            ax = plt.gca()

        means, stds = self.average_posterior_samples(
            variable_name=variable_name, return_std=True
        )

        time_steps = np.arange(means.shape[-1])
        if len(means.shape) == 1:
            # Coordination plot
            plot_series(
                x=time_steps,
                y=means,
                y_std=stds,
                label=kwargs.pop("label", None),
                include_bands=include_bands,
                value_bounds=value_bounds,
                ax=ax,
                **kwargs,
            )
            ax.set_ylabel("Coordination")
        elif len(means.shape) == 2:
            label_prefix = kwargs.pop("label", "Subject")
            # Serial variable
            subject_indices = np.array(
                [
                    int(x.split("#")[0])
                    for x in getattr(means, f"{variable_name}_time").data
                ]
            )
            if show_time_in_coordination_scale:
                time_steps = np.array(
                    [
                        int(x.split("#")[1])
                        for x in getattr(means, f"{variable_name}_time").data
                    ]
                )
            else:
                time_steps = np.arange(len(subject_indices))
            subjects = sorted(list(set(subject_indices)))
            for s in subjects:
                idx = [i for i, subject in enumerate(subject_indices) if subject == s]
                y = (
                    means.loc[dimension][idx]
                    if isinstance(dimension, str)
                    else means[dimension, idx]
                )
                y_std = (
                    stds.loc[dimension][idx]
                    if isinstance(dimension, str)
                    else stds[dimension, idx]
                ) if stds is not None else None
                plot_series(
                    x=time_steps[idx],
                    y=y,
                    y_std=y_std,
                    label=f"{label_prefix} {s}",
                    include_bands=include_bands,
                    value_bounds=value_bounds,
                    ax=ax,
                    **kwargs,
                )
            ax.set_ylabel(getattr(means, f"{variable_name}_dimension").data[dimension])
        else:
            label_prefix = kwargs.pop("label", "Subject")
            # Non-serial variable
            for s in range(means.shape[0]):
                y = (
                    means[s].loc[dimension]
                    if isinstance(dimension, str)
                    else means[s, dimension]
                )
                y_std = (
                    stds[s].loc[dimension]
                    if isinstance(dimension, str)
                    else stds[s, dimension]
                ) if stds is not None else None
                plot_series(
                    x=time_steps,
                    y=y,
                    y_std=y_std,
                    label=f"{label_prefix} {s}",
                    include_bands=include_bands,
                    value_bounds=value_bounds,
                    ax=ax,
                    **kwargs,
                )
            ax.set_ylabel(getattr(means, f"{variable_name}_dimension").data[dimension])

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
        return self.trace.sample_stats.lp.to_numpy()

    def get_posterior_samples(self, variable_name: str, sample_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return posterior samples from a variable.

        @param variable_name: name of the variable.
        @param sample_idx: optional samples to retrieve. If undefined, all samples will be
            returned.
        @return: posterior samples. The first dimension indexes the samples.
        """
        if variable_name in self.trace.observed_data:
            # Value was given, not sampled.
            return self.trace.observed_data[variable_name].to_numpy()
        else:
            values = self.trace.posterior[variable_name].stack(
                sample=["draw", "chain"]).transpose("sample", ...).to_numpy()

            if values.ndim == 1:
                values = values[:, None]

            if sample_idx is not None:
                values = values[sample_idx]

            return values
