from __future__ import annotations

from typing import Callable, Dict, List, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from coordination.common.constants import (DEFAULT_BURN_IN, DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS_PER_INFERENCE,
                                           DEFAULT_NUM_SAMPLED_SERIES,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED, DEFAULT_TARGET_ACCEPT)
from coordination.common.plot import plot_series
from coordination.inference.inference_data import InferenceData
from coordination.module.component_group import (ComponentGroup,
                                                 ComponentGroupSamples)
from coordination.module.coordination.coordination import Coordination
from coordination.module.module import Module, ModuleSamples


class Model(Module):
    """
    This class represents a coordination model comprised of a coordination module and several
    component groups with their associated latent components and observation variables.
    """

    def __init__(
        self,
        name: str,
        pymc_model: pm.Model,
        coordination: Coordination,
        component_groups: List[ComponentGroup],
    ):
        """
        Creates a model instance.

        @param name: a name for the model.
        @param pymc_model: a PyMC model instance where model's modules are to be created at.
        @param coordination: coordination module.
        @param component_groups: list of component groups in the model.
        """

        super().__init__(
            uuid=name, pymc_model=pymc_model, parameters=None, observed_values=None
        )

        self.coordination = coordination
        self.component_groups = component_groups

    def draw_samples(
        self,
        seed: Optional[int] = DEFAULT_SEED,
        num_series: int = DEFAULT_NUM_SAMPLED_SERIES,
    ) -> ModelSamples:
        """
        Draws samples from the model using ancestral sampling and some blending strategy with
        coordination and different subjects.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: model samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        coordination_samples = self.coordination.draw_samples(seed, num_series)

        # Place the latent component and observation samples from all groups in a single
        # dictionary for easy query.
        component_group_samples = {}
        for g in self.component_groups:
            g.latent_component.coordination_samples = coordination_samples
            group_samples = g.draw_samples(None, num_series)
            component_group_samples[
                g.latent_component.uuid
            ] = group_samples.latent_component_samples
            component_group_samples[
                f"common_cause_{g.latent_component.uuid}"
            ] = group_samples.common_cause_samples
            for obs_uuid, obs_samples in group_samples.observation_samples.items():
                component_group_samples[obs_uuid] = obs_samples

        return ModelSamples(coordination_samples, component_group_samples)

    def create_random_variables(self):
        """
        Creates random variables for coordination and component groups.
        """
        super().create_random_variables()

        self.coordination.create_random_variables()
        for g in self.component_groups:
            if g.common_cause:
                g.common_cause.coordination_random_variable = (
                    self.coordination.coordination_random_variable
                )
            g.latent_component.coordination_random_variable = (
                self.coordination.coordination_random_variable
            )
            g.create_random_variables()

    def fit(
        self,
        seed: Optional[int] = DEFAULT_SEED,
        burn_in: int = DEFAULT_BURN_IN,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        num_chains: int = DEFAULT_NUM_CHAINS,
        num_jobs: int = DEFAULT_NUM_JOBS_PER_INFERENCE,
        nuts_init_methods: str = DEFAULT_NUTS_INIT_METHOD,
        target_accept: float = DEFAULT_TARGET_ACCEPT,
        callback: Callable = None,
        **kwargs,
    ) -> InferenceData:
        """
        Performs inference in a model to estimate the latent variables posterior.

        @param seed: random seed for reproducibility.
        @param burn_in: number of samples to use as warm-up.
        @param num_samples: number of samples from the posterior distribution.
        @param num_chains: number of parallel chains.
        @param num_jobs: number of jobs (typically equals the number of chains)
        @param nuts_init_methods: initialization method of the NUTS algorithm.
        @param target_accept: target accept value. The higher, the smaller the number of
            divergences usually but it takes longer to converge.
        @param callback: functions to be called at every sample draw during model fit.
        @param: **kwargs: extra parameters to pass to the PyMC sample function.
        @return: inference data with posterior trace.
        """

        with self.pymc_model:
            idata = pm.sample(
                num_samples,
                init=nuts_init_methods,
                tune=burn_in,
                chains=num_chains,
                random_seed=seed,
                cores=min(num_jobs, num_chains),
                target_accept=target_accept,
                callback=callback,
                **kwargs,
            )

        return InferenceData(idata)

    def prior_predictive(
        self, seed: Optional[int] = DEFAULT_SEED, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> InferenceData:
        """
        Executes prior predictive checks in the model.

        @param seed: random seed for reproducibility.
        @param num_samples: number of samples from the posterior distribution.
        @return: inference data with prior checks.
        """

        with self.pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return InferenceData(idata)

    def posterior_predictive(
        self, posterior_trace: az.InferenceData, seed: Optional[int] = DEFAULT_SEED
    ) -> InferenceData:
        """
        Executes posterior predictive checks in the model.

        @param posterior_trace: inference data generate from a call to fit().
        @param seed: random seed for reproducibility.

        @return: inference data with posterior checks.
        """

        with self.pymc_model:
            idata = pm.sample_posterior_predictive(
                trace=posterior_trace, random_seed=seed
            )

        return InferenceData(idata)


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class ModelSamples(ModuleSamples):
    def __init__(
        self,
        coordination_samples: ModuleSamples,
        component_group_samples: Dict[str, ComponentGroupSamples],
    ):
        """
        Creates an object to store latent samples and samples from associates observations.

        @param coordination_samples: samples generated by the coordination module.
        @param component_group_samples: a dictionary of samples from component group indexed by the
            group's id.
        """
        super().__init__(values=None)

        self.coordination_samples = coordination_samples
        self.component_group_samples = component_group_samples

    def plot(
        self,
        variable_uuid: str,
        ax: Optional[plt.axis] = None,
        series_idx: int = 0,
        dimension_idx: int = 0,
        subject_transformation: Callable = None,
        **kwargs,
    ) -> plt.axis:
        """
        Plots the time series of samples.

        @param variable_uuid: variable to plot.
        @param ax: axis to plot on. It will be created if not provided.
        @param series_idx: index of the series of samples to plot.
        @param dimension_idx: index of the dimension axis to plot.
        @param subject_transformation: function called per subject (series, subject) to transform
            the subject series before plotting.
        @param kwargs: extra parameters to pass to the plot function.
        @return: plot axis.
        """

        if ax is None:
            plt.figure()
            ax = plt.gca()

        if variable_uuid == Coordination.UUID:
            samples = self.coordination_samples
        else:
            samples = self.component_group_samples.get(variable_uuid, None)

        if samples is None:
            raise ValueError(f"Found no samples for the variable ({variable_uuid}).")

        time_steps = np.arange(samples.num_time_steps)
        sampled_values = samples.values[series_idx]
        if len(sampled_values.shape) == 1:
            # Coordination plot
            plot_series(
                x=time_steps,
                y=sampled_values,
                y_std=None,
                label=None,
                include_bands=False,
                value_bounds=None,
                ax=ax,
                **kwargs,
            )
            ax.set_ylabel("Coordination")
        elif len(sampled_values.shape) == 2:
            # Serial variable
            subject_indices = samples.subject_indices[series_idx]
            time_steps = samples.time_steps_in_coordination_scale[series_idx]
            subjects = sorted(list(set(subject_indices)))
            for s in subjects:
                idx = [i for i, subject in enumerate(subject_indices) if subject == s]
                y = sampled_values[dimension_idx, idx]
                y = subject_transformation(y, s) if subject_transformation else y
                plot_series(
                    x=time_steps[idx],
                    y=y,
                    y_std=None,
                    label=f"Subject {s}",
                    include_bands=False,
                    value_bounds=None,
                    ax=ax,
                    **kwargs,
                )
            ax.set_xlabel("Dimension")
        else:
            # Non-serial variable
            for s in range(sampled_values.shape[0]):
                y = sampled_values[s, dimension_idx]
                y = subject_transformation(y, s) if subject_transformation else y
                plot_series(
                    x=time_steps,
                    y=y,
                    y_std=None,
                    label=f"Subject {s}",
                    include_bands=False,
                    value_bounds=None,
                    ax=ax,
                    **kwargs,
                )

            ax.set_xlabel("Dimension")

        ax.set_xlabel("Time Step")
        ax.spines[["right", "top"]].set_visible(False)

        return ax
