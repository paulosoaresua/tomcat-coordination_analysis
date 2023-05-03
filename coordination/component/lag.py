from typing import Any, List, Optional

import numpy as np
import pymc as pm
import pytensor as pt

from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, UniformDiscreteParameterPrior


class LagParameters:

    def __init__(self, max_lag: int):
        self.lag = Parameter(UniformDiscreteParameterPrior(-max_lag, max_lag))

    def clear_values(self):
        self.lag.value = None


class LagSamples:

    def __init__(self):
        self.values = np.array([])


class Lag:

    def __init__(self, uuid: str, max_lag: int):
        self.uuid = uuid
        self.max_lag = max_lag

        self.parameters = LagParameters(max_lag=max_lag)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self.lag_name
        ]

    @property
    def lag_name(self) -> str:
        return self.uuid

    def draw_samples(self, num_series: int, num_lags: int, seed: Optional[int] = None) -> LagSamples:
        set_random_seed(seed)

        samples = LagSamples()
        samples.values = np.random.randint(low=self.parameters.lag.prior.lower,
                                           high=self.parameters.lag.prior.upper,
                                           size=(num_series, num_lags))

        return samples

    def update_pymc_model(self, num_lags) -> Any:
        lag = pm.DiscreteUniform(self.lag_name,
                                 lower=self.parameters.lag.prior.lower,
                                 upper=self.parameters.lag.prior.upper,
                                 size=num_lags,
                                 observed=self.parameters.lag.value)

        return lag
