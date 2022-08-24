from typing import Optional

from scipy.stats import bernoulli, truncnorm
import numpy as np
import random


class CoordinationGenerator:
    """
    This class generates synthetic values for a latent coordination variable over time.
    """
    def __init__(self, num_time_steps: int):
        self._num_time_steps = num_time_steps

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cs = []
        for t in range(self._num_time_steps):
            if t == 0:
                cs.append(self._sample_from_prior())
            else:
                cs.append(self._sample_from_transition(cs[t-1]))

        return np.array(cs)

    def _sample_from_prior(self) -> float:
        raise Exception("Not implemented in this class.")

    def _sample_from_transition(self, previous_coordination: float) -> float:
        raise Exception("Not implemented in this class.")


class DiscreteCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a binary-variable coordination.
    """

    def __init__(self, p_coordinated: float, p_transition: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prior = bernoulli(p_coordinated)
        self._transition = bernoulli(p_transition)

    def _sample_from_prior(self) -> float:
        return self._prior.rvs()

    def _sample_from_transition(self, previous_coordination: float) -> float:
        return 1 - previous_coordination if self._transition.rvs() == 1 else previous_coordination


class ContinuousCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a continuous coordination.
    """

    MIN_VALUE = 0
    MAX_VALUE = 1

    def __init__(self, prior_mean: float, prior_std: float, transition_std: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert ContinuousCoordinationGenerator.MIN_VALUE <= prior_mean <= ContinuousCoordinationGenerator.MAX_VALUE

        self._transition_std = transition_std

        self._prior = truncnorm(loc=prior_mean, scale=prior_std, a=0, b=1)

    def _sample_from_prior(self) -> float:
        return self._prior.rvs()

    def _sample_from_transition(self, previous_coordination: float) -> float:
        a = (ContinuousCoordinationGenerator.MIN_VALUE - previous_coordination) / self._transition_std
        b = (ContinuousCoordinationGenerator.MAX_VALUE - previous_coordination) / self._transition_std
        transition = truncnorm(loc=previous_coordination, scale=self._transition_std, a=a, b=b)

        return transition.rvs()


