import numpy as np
from scipy.stats import norm

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as at

from coordination.common.distribution import beta
from coordination.common.utils import logit, sigmoid

T = 50
N = 1
S = 3
C = 2

SUC = 0.1
SC = 0.1
SA = 1
SAA = 1
SO = 1
W = np.array([0.5, 0.5])

MIN_C = 1e-6
MAX_C = 1 - 1e-6

if __name__ == "__main__":
    np.random.seed(0)

    T = 50

    x = norm(loc=logit(0.5), scale=0.1).rvs(T).cumsum()
    y = norm(loc=sigmoid(x[None, :]), scale=0.1).rvs((4, T))
    z = norm(loc=y, scale=0.1).rvs()

    with pm.Model() as model:
        sdx = pm.HalfNormal("sdx", sigma=1, shape=1)
        sdy = pm.HalfNormal("sdy", sigma=1, shape=1)
        sdz = pm.HalfNormal("sdz", sigma=1, shape=1)

        xv = pm.GaussianRandomWalk("xv", mu=0, sigma=sdx, shape=T)
        tmp = pm.Deterministic("tmp", pm.math.sigmoid(xv))
        yv = pm.Normal("yv", mu=tmp[None, :], sigma=sdy, shape=(4, T))
        zv = pm.Normal("zv", mu=yv, sigma=sdz, shape=(4, T), observed=z)

        idata = pm.sample_prior_predictive(1000, random_seed=0)

        prior_obs = idata.prior_predictive["zv"].sel(chain=0).to_numpy()

        for i in range(4):
            plt.figure(figsize=(15, 8))
            plt.plot(np.arange(T)[None, :].repeat(1000, axis=0).T, prior_obs[:, i].T, label="Predicted", color="tab:blue", alpha=0.3)
            plt.plot(np.arange(T), z[i], label="Real", color="black", marker="o")
            plt.title(f"Prior Predictive. Feature {i}")
            plt.show()

        idata = pm.sample(1000, init="adapt_diag", tune=1000, chains=2, random_seed=0)

        az.plot_trace(idata, var_names=["sdx", "sdy", "sdz"])
        plt.show()

        m1 = idata.posterior["xv"].mean(dim=["chain", "draw"])
        sd1 = idata.posterior["xv"].std(dim=["chain", "draw"])

        plt.figure(figsize=(15, 8))
        plt.plot(range(T), x, label="Real", color="tab:blue", marker="o")
        plt.plot(range(T), m1, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), m1 - sd1, m1 + sd1, color="tab:orange", alpha=0.4)
        plt.title("X")
        plt.legend()
        plt.show()

        m1 = idata.posterior["yv"].sel(yv_dim_0=0).mean(dim=["chain", "draw"])
        sd1 = idata.posterior["yv"].sel(yv_dim_0=0).std(dim=["chain", "draw"])

        plt.figure(figsize=(15, 8))
        plt.plot(range(T), y[0], label="Real", color="tab:blue", marker="o")
        plt.plot(range(T), m1, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), m1 - sd1, m1 + sd1, color="tab:orange", alpha=0.4)
        plt.title("Y")
        plt.legend()
        plt.show()
