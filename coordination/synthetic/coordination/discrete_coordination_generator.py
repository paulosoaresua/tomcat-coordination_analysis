from scipy.stats import bernoulli

from coordination.synthetic.coordination.coordination_generator import CoordinationGenerator


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

