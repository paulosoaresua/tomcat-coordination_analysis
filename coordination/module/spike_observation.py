from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import bernoulli

from coordination.common.utils import set_random_seed
from coordination.module.parametrization import BetaParameterPrior, Parameter


class SpikeObservationParameters:
    def __init__(self, a_p: float, b_p: float):
        self.p = Parameter(BetaParameterPrior(a_p, b_p))

    def clear_values(self):
        self.p.value = None


class SpikeObservationSamples:
    def __init__(self):
        # For each time step in the component's scale, it contains the time step in the
        # coordination scale were a spike occurred.
        self.time_steps_in_coordination_scale: List[np.ndarray] = []

    @property
    def num_time_steps(self):
        if len(self.time_steps_in_coordination_scale) == 0:
            return 0

        return self.time_steps_in_coordination_scale[0].shape[-1]


class SpikeObservation:
    """
    This class models semantic links or any kind of binary observations with similar distribution.
    """

    def __init__(self, uuid: str, a_p: float, b_p: float):
        self.uuid = uuid

        self.parameters = SpikeObservationParameters(a_p, b_p)

    @property
    def parameter_names(self) -> List[str]:
        return [self.p_name]

    @property
    def p_name(self) -> str:
        return f"p_{self.uuid}"

    def draw_samples(
        self,
        num_series: int,
        time_scale_density: float,
        coordination: np.ndarray,
        seed: Optional[int] = None,
    ) -> SpikeObservationSamples:
        set_random_seed(seed)

        samples = SpikeObservationSamples()

        # Randomly sample candidate time steps in which we observe a link
        density_mask = bernoulli(p=time_scale_density).rvs(coordination.shape)

        # Effectively observe links according to the values of coordination
        links = bernoulli(p=coordination * self.parameters.p.value).rvs(
            coordination.shape
        )

        links *= density_mask

        for s in range(num_series):
            # We don't have numerical values but time steps when links are observed
            samples.time_steps_in_coordination_scale.append(
                np.array([t for t, l in enumerate(links[s]) if l == 1])
            )

        return samples

    def update_pymc_model(
        self, coordination: Any, time_dimension: str, observed_values: Any
    ) -> Any:
        p = pm.Beta(
            name=self.p_name,
            alpha=self.parameters.p.prior.a,
            beta=self.parameters.p.prior.b,
            size=1,
            observed=self.parameters.p.value,
        )

        adjusted_prob = pm.Deterministic(f"adjusted_prob_{self.uuid}", p * coordination)

        pm.Bernoulli(
            self.uuid, adjusted_prob, dims=time_dimension, observed=observed_values
        )

        return p, adjusted_prob
