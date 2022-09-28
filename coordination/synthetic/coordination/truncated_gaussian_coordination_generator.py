from scipy.stats import truncnorm

from coordination.synthetic.coordination.coordination_generator import CoordinationGenerator


class TruncatedGaussianCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a continuous coordination sampled from a truncated Gaussian distribution
    to constrain the samples between 0 and 1.
    """

    MIN_VALUE = 0
    MAX_VALUE = 1

    def __init__(self, mean_prior_coordination: float, std_prior_coordination: float, std_coordination_drifting: float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert TruncatedGaussianCoordinationGenerator.MIN_VALUE <= mean_prior_coordination \
               <= TruncatedGaussianCoordinationGenerator.MAX_VALUE

        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting

    def _sample_from_prior(self) -> float:
        if self._std_prior_coordination == 0:
            return self._mean_prior_coordination
        else:
            lb = TruncatedGaussianCoordinationGenerator.MIN_VALUE
            ub = TruncatedGaussianCoordinationGenerator.MAX_VALUE
            a = (lb - self._mean_prior_coordination) / self._std_prior_coordination
            b = (ub - self._mean_prior_coordination) / self._std_prior_coordination

        return truncnorm.rvs(loc=self._mean_prior_coordination, scale=self._std_prior_coordination, a=a, b=b)

    def _sample_from_transition(self, previous_coordination: float) -> float:
        a = (TruncatedGaussianCoordinationGenerator.MIN_VALUE - previous_coordination) / self._std_coordination_drifting
        b = (TruncatedGaussianCoordinationGenerator.MAX_VALUE - previous_coordination) / self._std_coordination_drifting
        transition = truncnorm(loc=previous_coordination, scale=self._std_coordination_drifting, a=a, b=b)

        return transition.rvs()
