from abc import abstractmethod
from typing import Callable, Optional

import arviz as az
import numpy as np
import pymc as pm
import pandas as pd

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
from coordination.module.serial_observation import SerialObservation
from coordination.module.non_serial_observation import NonSerialObservation


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

        self._create_model_from_config_bundle()

    @property
    def num_time_steps_in_coordination_scale(self) -> int:
        """
        Gets the number of time steps in coordination scale to fit. We get this value from the
        coordination module instead of the config bundle because they may be different. The config
        bundle is just a guide one may choose to parameterize time steps differently given other
        options in the bundle.

        @return: Number of time steps in coordination scale to fit.
        """
        return self._model.coordination.num_time_steps

    @abstractmethod
    def _register_metadata(self, config_bundle: ModelConfigBundle):
        """
        Add entries to the metadata dictionary from values filled in the config bundle.

        @param config_bundle: config bundle to fill up the metadata with.
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

    def new_config_bundle_from_time_step_info(
            self,
            config_bundle: ModelConfigBundle) -> ModelConfigBundle:
        """
        Gets a new config bundle with metadata and observed values adapted to the number of time
        steps in coordination scale in case we don't want to fit just a portion of the time series.

        @param config_bundle: original config bundle.
        @return: new config bundle.
        """
        self._register_metadata(config_bundle)

        new_bundle = deepcopy(config_bundle)
        if new_bundle.num_time_steps_to_fit is None:
            new_bundle.num_time_steps_to_fit = new_bundle.num_time_steps_in_coordination_scale

        # Update metadata. This will adjust the metadata indices to make sure we do not include
        # any time step that is not being fit/sampled.
        for key, meta in self.metadata.items():
            self.metadata[key] = meta.truncate(new_bundle.num_time_steps_to_fit)

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

    def get_ppa_summary(self,
                        idata: InferenceData,
                        window_size: int,
                        num_samples: int,
                        seed: int) -> pd.DataFrame:
        """
        Initializes the model with a subset of samples from the posterior distribution and
        performs ancestral sampling up to a time window. Then, we compute the MSE of the generated
        samples and the observations in this window. We also report the MSE of two baseline models
        fit on previous observed data to predict the observed data in the window: an average model
        and a linear model.

        @param idata: inference data.
        @param window_size: how long in the future to generate samples.
        @param num_samples: number of samples from posterior to use. Samples will be chosen
            randomly from the posterior samples.
        @param seed: random seed for reproducibility when choosing the samples to keep.
        @return: a dataframe with summarized results.
        """
        original_bundle = self.config_bundle
        self.config_bundle = self.new_config_bundle_from_posterior_samples(
            config_bundle=original_bundle,
            idata=idata,
            num_samples=num_samples,
            seed=seed)

        lb = self.config_bundle.num_time_steps_to_fit
        ub = lb + window_size

        self.config_bundle.num_time_steps_to_fit = ub
        samples = self.draw_samples(num_series=num_samples)
        results = []
        for g in self._model.component_groups:
            for o in g.observations:
                if (not isinstance(o, SerialObservation) and
                        not isinstance(o, NonSerialObservation)):
                    # We only compute MSE for observations that generate real values.
                    continue

                # Prediction and real data in the prediction window
                real_data = self.metadata[o.uuid].normalized_observations[..., lb:ub]
                pred_data = np.mean(samples.component_group_samples[o.uuid].values, axis=0)[...,
                            lb:ub]

                mse = np.square(real_data - pred_data) / np.arange(1, window_size + 1)
                # Compute the mse across all the dimensions but the last one (the window)
                mse = mse.mean(axis=mse.shape[:-1])

                results.append({
                    "model": self._model.uuid,
                    "variable": o.uuid,
                    **{f"w{w}": mse[w - 1] for w in range(1, window_size + 1)}
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    from coordination.model.config_bundle.vocalic import VocalicConfigBundle
    from coordination.model.real.vocalic import VocalicModel
    import pickle
    from coordination.inference.inference_data import InferenceData

    bundle = VocalicConfigBundle()
    bundle.num_time_steps_to_fit = 78

    model = VocalicModel(config_bundle=bundle)

    with open("../../.run/inferences/ppa_true/T000612/inference_data.pkl", "rb") as f:
        idata = InferenceData(pickle.load(f))
        model.get_ppa_summary(idata, 5, 100, 0)
