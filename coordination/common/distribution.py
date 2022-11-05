import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm as scipy_truncnorm


class TruncatedNormal:

    def __init__(self, mean: np.ndarray, std: np.ndarray, a: float = 0, b: float = 1):
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

        alpha = (self.a - self.mean) / self.std
        beta = (self.b - self.mean) / self.std
        self.offset = (norm().pdf(alpha) - norm.pdf(beta)) * self.std / (norm.cdf(beta) - norm.cdf(alpha))
        self._scipy_distribution = scipy_truncnorm(loc=mean, scale=std, a=alpha, b=beta)

    def rvs(self) -> np.ndarray:
        return self._rejection_sampling(self.mean, self.std)

    def _rejection_sampling(self, mean: np.ndarray, std: np.ndarray):
        proposal = norm(mean, std)
        samples = proposal.rvs()

        out_of_bounds_indices = (samples < self.a) | (samples > self.b)
        if np.any(out_of_bounds_indices):
            if isinstance(samples, np.ndarray):
                samples[out_of_bounds_indices] = self._rejection_sampling(mean[out_of_bounds_indices],
                                                                          std[out_of_bounds_indices])
            else:
                samples = self._rejection_sampling(mean, std)

        return samples

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_distribution.logpdf(x)


if __name__ == "__main__":
    NUM_SAMPLES = 10000
    TIME_STEPS = 100

    import time

    start = time.time()
    dist = TruncatedNormal(np.zeros((NUM_SAMPLES, 1)), np.ones((NUM_SAMPLES, 1)) * 0.1)
    samples = dist.rvs()
    print((time.time() - start))

    start = time.time()
    dist._scipy_distribution.rvs()
    print((time.time() - start))

    start = time.time()
    norm(np.zeros((NUM_SAMPLES, 1)), np.ones((NUM_SAMPLES, 1)) * 0.1).rvs()
    print((time.time() - start))

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(samples, bins=100)
    plt.show()

    print(f"Mean = {samples.mean()}")
    print(f"Variance = {samples.var()}")

    from scipy.stats import lognorm, beta

    # Transition
    std = np.ones((NUM_SAMPLES, 1)) * 0.05
    lognorm_mean = np.exp(np.ones((NUM_SAMPLES, 2))) * [3e3, 7e3]
    mean = np.ones((NUM_SAMPLES, 2)) * 0.1
    means = [0.3]
    logprobs = []
    css = []
    css_std = []
    chain = []
    for t in range(TIME_STEPS):
        dist = lognorm(loc=0, scale=lognorm_mean, s=std)
        lognorm_mean = dist.rvs()
        cs = beta(a=lognorm_mean[:, 0], b=lognorm_mean[:, 1]).rvs()

        # dist = norm(mean, std)
        # cs = dist.rvs()

        # logprobs.append(dist.logpdf(mean[:, 0]).mean())
        css.append(cs.mean())
        css_std.append(cs.std())
        chain.append(cs)
        # means.append(mean.mean())

    css = np.array(css)
    css_std = np.array(css_std)
    chain = np.array(chain).T

    # plt.figure()
    # plt.plot(range(51), means, color="tab:blue", alpha=0.7, marker="o")
    # plt.ylim([0, 1])
    # plt.title("MS")
    # plt.show()

    plt.figure()
    plt.plot(range(TIME_STEPS), css, color="tab:blue", alpha=0.7, marker="o", markersize=3)
    plt.fill_between(range(TIME_STEPS), css - css_std, css + css_std, color="tab:blue", alpha=0.4)
    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        plt.plot(range(TIME_STEPS), np.ones(TIME_STEPS) * i, color="tab:red", alpha=0.7, linestyle="--")
    plt.ylim([0, 1])
    plt.title("CS")
    plt.show()

    plt.figure()
    for i in range(5):
        plt.plot(range(TIME_STEPS), chain[i], alpha=0.7, marker="o", markersize=3)
    plt.ylim([0, 1])
    plt.title("Chain")
    plt.show()

    # plt.figure()
    # plt.plot(range(50), logprobs, color="tab:orange", alpha=0.7, marker="o")
    # plt.ylim([0, 1])
    # plt.title("LL")
    # plt.show()
