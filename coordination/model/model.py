from __future__ import annotations
from typing import Dict, List, Optional

import pymc as pm

from coordination.module.coordination.coordination import Coordination
from coordination.module.component_group import ComponentGroup, ComponentGroupSamples
from coordination.module.module import Module, ModuleSamples
from coordination.module.constants import (DEFAULT_SEED,
                                           DEFAULT_NUM_SAMPLED_SERIES,
                                           DEFAULT_BURN_IN,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS,
                                           DEFAULT_INIT_METHOD,
                                           DEFAULT_TARGET_ACCEPT)
from coordination.entity.inference_data import InferenceData


class Model(Module):
    """
    This class represents a coordination model comprised of a coordination module and several
    component groups with their associated latent components and observation variables.
    """

    def __init__(self,
                 name: str,
                 pymc_model: pm.Model,
                 coordination: Coordination,
                 component_groups: List[ComponentGroup],
                 coordination_samples: Optional[ModuleSamples] = None):
        """
        Creates a model instance.

        @param name: a name for the model.
        @param pymc_model: a PyMC model instance where model's modules are to be created at.
        @param coordination: coordination module.
        @param component_groups: list of component groups in the model.
        @param coordination_samples: fixed coordination samples to be used during a call to
            draw_samples. If provided, these samples will be used and samples from the coordination
            component won't be drawn.
        """

        super().__init__(
            uuid=name,
            pymc_model=pymc_model,
            parameters=None,
            observed_values=None
        )

        self.coordination = coordination
        self.component_groups = component_groups
        self.coordination_samples = coordination_samples

    def draw_samples(self,
                     seed: Optional[int] = DEFAULT_SEED,
                     num_series: int = DEFAULT_NUM_SAMPLED_SERIES) -> ModelSamples:
        """
        Draws samples from the model using ancestral sampling and some blending strategy with
        coordination and different subjects.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: model samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if self.coordination_samples is None:
            coordination_samples = self.coordination.draw_samples(seed, num_series)
        else:
            coordination_samples = self.coordination_samples
        component_group_samples = {}
        for g in self.component_groups:
            g.latent_component.coordination_samples = coordination_samples
            component_group_samples[g.uuid] = g.draw_samples(seed, num_series)

        return ModelSamples(coordination_samples, component_group_samples)

    def create_random_variables(self):
        """
        Creates random variables for coordination and component groups.
        """
        super().create_random_variables()

        self.coordination.create_random_variables()
        for g in self.component_groups:
            g.latent_component.coordination_random_variable = \
                self.coordination.coordination_random_variable
            g.create_random_variables()

    def fit(self,
            seed: Optional[int] = DEFAULT_SEED,
            burn_in: int = DEFAULT_BURN_IN,
            num_samples: int = DEFAULT_NUM_SAMPLES,
            num_chains: int = DEFAULT_NUM_CHAINS,
            num_jobs: int = DEFAULT_NUM_JOBS,
            init_method: str = DEFAULT_INIT_METHOD,
            target_accept: float = DEFAULT_TARGET_ACCEPT,
            **kwargs) -> InferenceData:
        """
        Performs inference in a model to estimate the latent variables posterior.

        @param seed: random seed for reproducibility.
        @param burn_in: number of samples to use as warm-up.
        @param num_samples: number of samples from the posterior distribution.
        @param num_chains: number of parallel chains.
        @param num_jobs: number of jobs (typically equals the number of chains)
        @param init_method: initialization method.
        @param target_accept: target accept value. The higher, the smaller the number of
            divergences usually but it takes longer to converge.
        @param: **kwargs: extra parameters to pass to the PyMC sample function.
        @return: inference data with posterior trace.
        """

        with self.pymc_model:
            idata = pm.sample(num_samples,
                              init=init_method,
                              tune=burn_in,
                              chains=num_chains,
                              random_seed=seed,
                              cores=num_jobs,
                              target_accept=target_accept,
                              **kwargs)

        return InferenceData(idata)

    def prior_predictive(self,
                         seed: Optional[int] = DEFAULT_SEED,
                         num_samples: int = DEFAULT_NUM_SAMPLES) -> az.InferenceData:
        """
        Executes prior predictive checks in the model.

        @param seed: random seed for reproducibility.
        @param num_samples: number of samples from the posterior distribution.
        @return: inference data with prior checks.
        """

        with self.pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return idata

    def posterior_predictive(self,
                             posterior_trace: az.InferenceData,
                             seed: Optional[int] = DEFAULT_SEED) -> az.InferenceData:
        """
        Executes posterior predictive checks in the model.

        @param posterior_trace: inference data generate from a call to fit().
        @param seed: random seed for reproducibility.

        @return: inference data with posterior checks.
        """

        with self.pymc_model:
            idata = pm.sample_posterior_predictive(trace=posterior_trace, random_seed=seed)

        return idata


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class ModelSamples(ModuleSamples):

    def __init__(self,
                 coordination_samples: ModuleSamples,
                 component_group_samples: Dict[str, ComponentGroupSamples]):
        """
        Creates an object to store latent samples and samples from associates observations.

        @param coordination_samples: samples generated by the coordination module.
        @param component_group_samples: a dictionary of samples from component group indexed by the
            group's id.
        """
        super().__init__(values=None)

        self.coordination_samples = coordination_samples
        self.component_group_samples = component_group_samples
