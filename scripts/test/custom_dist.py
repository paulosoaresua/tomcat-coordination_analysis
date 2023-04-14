from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as ptt


def logp_fn(x: Any,
            mu: Any,
            sigma: Any,
            dummy: Any):
    return pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), x).sum()


if __name__ == "__main__":
    evidence = np.random.normal(loc=3, size=1000)

    with pm.Model() as model:
        logp_params = [
            ptt.ones(1) * 3,
            ptt.ones(1),
            pm.Normal("dummy", mu=0, sigma=1, size=(5,))
        ]
        latent = pm.DensityDist("custom", *logp_params, logp=logp_fn, size=(1,))

        obs = pm.Normal("observation", mu=latent, sigma=1, observed=evidence)

        idata = pm.sample(1000, init="jitter+adapt_diag", tune=1000, chains=2, random_seed=0, cores=2)
        az.plot_trace(idata)
        plt.show()
