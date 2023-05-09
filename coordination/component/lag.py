from typing import Any, List

import pymc as pm

from coordination.model.parametrization import Parameter, UniformDiscreteParameterPrior


class LagParameters:

    def __init__(self, max_lag: int):
        self.lag = Parameter(UniformDiscreteParameterPrior(-max_lag, max_lag))

    def clear_values(self):
        self.lag.value = None


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

    def clear_parameter_values(self):
        self.parameters.clear_values()

    def update_pymc_model(self, num_lags) -> Any:
        lag = pm.DiscreteUniform(self.lag_name,
                                 lower=self.parameters.lag.prior.lower,
                                 upper=self.parameters.lag.prior.upper,
                                 size=num_lags,
                                 observed=self.parameters.lag.value)

        return lag
