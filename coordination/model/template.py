from abc import abstractmethod
from typing import Callable, Optional

import arviz as az
import pymc as pm

from coordination.common.constants import (DEFAULT_BURN_IN, DEFAULT_NUM_CHAINS,
                                           DEFAULT_NUM_JOBS_PER_INFERENCE,
                                           DEFAULT_NUM_SAMPLED_SERIES,
                                           DEFAULT_NUM_SAMPLES,
                                           DEFAULT_NUTS_INIT_METHOD,
                                           DEFAULT_SEED, DEFAULT_TARGET_ACCEPT)
from coordination.inference.inference_data import InferenceData
from coordination.model.config_bundle.bundle import ModelConfigBundle
from coordination.model.model import Model, ModelSamples
from copy import deepcopy
from coordination.metadata.metadata import Metadata


class ModelTemplate:
    """
    This class represents a template for concrete models. It extends the model class and
    incorporates helper functions to be called to set specific parameters needed for sampling and
    inference.
    """

    def __init__(self, pymc_model: pm.Model, config_bundle: ModelConfigBundle):
        """
        Creates a model template instance.

        @param pymc_model: a PyMC model instance where model's modules are to be created at.
        @param config_bundle: a config bundle with values for the different parameters of the
            model.
        """

        if not pymc_model:
            self.pymc_model = pm.Model()

        self.config_bundle = config_bundle

        self._model: Model = None
        self.metadata: Dict[str, Metadata] = {}
        self._register_metadata()

        # Adjust metadata to the adjusted time step info. We don't replace the original config
        # bundle with the adjusted one because we want to preserve the original one.
        self.new_config_bundle_from_time_step_info(self.config_bundle)

    @abstractmethod
    def _register_metadata(self):
        """
        Add entries to the metadata dictionary from values filled in the config bundle.
        """
        pass

    @abstractmethod
    def _create_model_from_config_bundle(self):
        """
        Creates internal modules of the model using the most up-to-date information in the config
        bundle. This allows the config bundle to be updated after the model creation, reflecting
        in changes in the model's modules any time this function is called.
        """
        pass

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
        self._create_model_from_config_bundle()
        return self._model.draw_samples(seed, num_series)

    def prepare_for_inference(self):
        """
        Constructs the model from the most-up-to-date information in the config bundle to be used
        with any of the inference methods: prior_predictive, fit or posterior_predictive. This
        avoids the need to create a new model at the call to these methods since they should be
        using the same underlying parameters.
        """
        self._create_model_from_config_bundle()
        self._model.create_random_variables()

    def prior_predictive(
            self, seed: Optional[int] = DEFAULT_SEED, num_samples: int = DEFAULT_NUM_SAMPLES
    ) -> InferenceData:
        """
        Executes prior predictive checks in the model.

        @param seed: random seed for reproducibility.
        @param num_samples: number of samples from the posterior distribution.
        @return: inference data with prior checks.
        """
        if self._model is None:
            raise ValueError(
                "The underlying model structure is undefined. Make sure to call the method "
                "prepare_for_inference before calling any of the inference methods."
            )

        return self._model.prior_predictive(seed=seed, num_samples=num_samples)

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
        if self._model is None:
            raise ValueError(
                "The underlying model structure is undefined. Make sure to call the method "
                "prepare_for_inference before calling any of the inference methods."
            )

        return self._model.fit(
            seed=seed,
            burn_in=burn_in,
            num_samples=num_samples,
            num_chains=num_chains,
            num_jobs=num_jobs,
            nuts_init_methods=nuts_init_methods,
            target_accept=target_accept,
            callback=callback,
            **kwargs,
        )

    def posterior_predictive(
            self, posterior_trace: az.InferenceData, seed: Optional[int] = DEFAULT_SEED
    ) -> InferenceData:
        """
        Executes posterior predictive checks in the model.

        @param posterior_trace: inference data generate from a call to fit().
        @param seed: random seed for reproducibility.

        @return: inference data with posterior checks.
        """

        if self._model is None:
            raise ValueError(
                "The underlying model structure is undefined. Make sure to call the method "
                "prepare_for_inference before calling any of the inference methods."
            )

        return self._model.posterior_predictive(
            posterior_trace=posterior_trace, seed=seed
        )

    @abstractmethod
    def new_config_bundle_from_time_step_info(
            self,
            config_bundle: ModelConfigBundle) -> ModelConfigBundle:
        """
        Gets a new config bundle with metadata and observed values adapted to the number of time
        steps in coordination scale in case we don't want to fit just a portion of the time series.

        @param config_bundle: original config bundle.
        @return: new config bundle.
        """

        new_bundle = deepcopy(config_bundle)
        if new_bundle.num_time_steps_to_fit is not None:
            num_time_steps_to_fit = new_bundle.num_time_steps_to_fit
        else:
            num_time_steps_to_fit = int(
                new_bundle.num_time_steps_in_coordination_scale *
                config_bundle.perc_time_steps_to_fit)

        new_bundle.num_time_steps_in_coordination_scale = num_time_steps_to_fit

        # Update metadata
        for key, meta in self.metadata.items():
            self.metadata[key] = meta.truncate(num_time_steps_to_fit)

        return new_bundle

    @abstractmethod
    def new_config_bundle_from_posterior_samples(self,
                                                 config_bundle: ModelConfigBundle,
                                                 idata: InferenceData,
                                                 num_samples: int,
                                                 seed: int) -> ModelConfigBundle:
        """
        Uses samples from posterior to update a config bundle. Here we set the samples from the
        posterior in the last time step as initial values for the latent variables. This
        allows us to generate samples in the future for predictive checks.

        @param config_bundle: original config bundle.
        @param idata: inference data.
        @param num_samples: number of samples from posterior to use. Samples will be chosen
            randomly from the posterior samples.
        @param seed: random seed for reproducibility when choosing the samples to keep.
        """
        pass
