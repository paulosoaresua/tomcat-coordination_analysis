from typing import Any, Callable, Dict, List, Tuple
from scipy.stats import bernoulli, truncnorm


class CoordinationGenerator:
    """
    This class generates synthetic values for a latent coordination variable over time.
    """

    def generate_evidence(self, time_steps: int) -> List[float]:
        cs = []
        for t in range(time_steps):
            if t == 0:
                cs.append(self._sample_from_prior())
            else:
                cs.append(self._sample_from_transition(cs[t-1]))

        return cs

    def _sample_from_prior(self) -> float:
        raise Exception("Not implemented in this class.")

    def _sample_from_transition(self, previous_c: float) -> float:
        raise Exception("Not implemented in this class.")


class DiscreteCoordinationGeneratorASIST(CoordinationGenerator):
    """
    This class generates synthetic values for a binary-variable coordination.
    """

    def __init__(self, p_prior: float, pc: float):
        self.__p_prior = p_prior
        self.__pc = pc

    def _sample_from_prior(self) -> float:
        return bernoulli.rvs(self.__p_prior)

    def _sample_from_transition(self, previous_c: float) -> float:
        if bernoulli.rvs(self.__pc) == 1:
            # Repeat the state
            return previous_c
        else:
            # Flip the state
            return 1 - previous_c


class DiscreteCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a binary-variable coordination.
    """

    def __init__(self, p_prior: float, p_transition: float):
        self._p_prior = p_prior
        self._p_transition = p_transition

    def _sample_from_prior(self) -> float:
        return bernoulli.rvs(self._p_prior)

    def _sample_from_transition(self, previous_c: float) -> float:
        if bernoulli.rvs(self._p_transition) == 1:
            # Flip the state
            return 1 - previous_c
        else:
            # Repeat the state
            return previous_c


class ContinuousCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a continuous coordination.
    """

    def _sample_from_prior(self) -> float:
        return 0

    def _sample_from_transition(self, previous_c: float) -> float:
        std = 0.1
        return truncnorm.rvs((0 - previous_c)/std, (1-previous_c)/std, loc=previous_c, scale=std)


