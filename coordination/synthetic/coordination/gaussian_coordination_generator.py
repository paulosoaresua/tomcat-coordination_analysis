from scipy.stats import truncnorm

from coordination.synthetic.coordination.coordination_generator import CoordinationGenerator


class GaussianCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a continuous coordination sampled from a truncated Gaussian distribution
    to constrain the samples between 0 and 1.
    """

    MIN_VALUE = 0
    MAX_VALUE = 1

    def __init__(self, prior_mean: float, prior_std: float, transition_std: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert GaussianCoordinationGenerator.MIN_VALUE <= prior_mean <= GaussianCoordinationGenerator.MAX_VALUE

        self._prior_mean = prior_mean
        self._prior_std = prior_std
        self._transition_std = transition_std

    def _sample_from_prior(self) -> float:
        if self._prior_std == 0:
            return self._prior_mean
        else:
            a = (GaussianCoordinationGenerator.MIN_VALUE - self._prior_mean) / self._prior_std
            b = (GaussianCoordinationGenerator.MAX_VALUE - self._prior_mean) / self._prior_std

        return truncnorm.rvs(loc=self._prior_mean, scale=self._prior_std, a=a, b=b)

    def _sample_from_transition(self, previous_coordination: float) -> float:
        a = (GaussianCoordinationGenerator.MIN_VALUE - previous_coordination) / self._transition_std
        b = (GaussianCoordinationGenerator.MAX_VALUE - previous_coordination) / self._transition_std
        transition = truncnorm(loc=previous_coordination, scale=self._transition_std, a=a, b=b)

        return transition.rvs()
