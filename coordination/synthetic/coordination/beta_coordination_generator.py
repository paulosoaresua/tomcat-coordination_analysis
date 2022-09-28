from scipy.stats import beta

from coordination.synthetic.coordination.coordination_generator import CoordinationGenerator


class BetaCoordinationGenerator(CoordinationGenerator):
    """
    This class generates synthetic values for a continuous coordination sampled from a beta distribution.
    """

    def __init__(self, prior_a: float, prior_b: float, transition_std: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._prior = beta(prior_a, prior_b)
        self._transition_std = transition_std

    def _get_transition_distribution(self, previous_coordination: float):
        """
        The transition distribution is a Beta distribution with mean equals to the previous coordination and
        fixed variance. Therefore, the parameters a and b of the resulting distribution depends on the previous
        coordination and fixed variance and can be determined analytically by solving the system of equations that
        define the mean and variance of a beta distribution:

        mean (= previous coordination) = a / (a + b)
        variance = (a + b)^2(a + b + 1)

        See section Mean and Variance:
        https://en.wikipedia.org/wiki/Beta_distribution
        """
        var = self._transition_std ** 2
        x = previous_coordination * (1 - previous_coordination)

        if var < x:
            b = previous_coordination * (
                    (1 - previous_coordination) / self._transition_std) ** 2 + previous_coordination - 1
            a = b * previous_coordination / (1 - previous_coordination)
            distribution = beta(a, b)
        elif previous_coordination < 0.1:
            distribution = beta(0.1, 1)
        elif previous_coordination > 0.9:
            distribution = beta(1, 0.1)
        else:
            raise Exception(f"Unsupported: {previous_coordination}")
        return distribution

    def _sample_from_prior(self) -> float:
        return self._prior.rvs()

    def _sample_from_transition(self, previous_coordination: float) -> float:
        return self._get_transition_distribution(previous_coordination).rvs()
