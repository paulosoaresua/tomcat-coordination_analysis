import pymc3 as pm
import arviz as az

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    T = 50
    F = 2

    sd_z = 1
    sd_x = 1

    np.random.seed(0)

    x = np.zeros((T, F))
    z = np.zeros((T, F))

    for t in range(T):
        if t == 0:
            z[0] = norm(loc=0, scale=1).rvs(size=F)
        else:
            z[t] = norm(loc=z[t - 1], scale=sd_z).rvs()

        x[t] = norm(loc=z[t], scale=sd_x).rvs()

    coords = {"time": np.arange(T), "features": np.arange(F)}
    hmm = pm.Model(coords=coords)

    with hmm:
        x_obs = pm.Data("x_obs", x, dims=["time", "features"])

        # Standard deviation of the transition and emission distributions
        i_sd_z = pm.HalfNormal(name="i_sd_z", sigma=1)
        i_sd_x = pm.HalfNormal(name="i_sd_x", sigma=1)

        # Transition distribution of latent variable Z
        # i_z_0 = pm.Normal(name='i_z_0', sigma=1, dims=['features'])
        # i_z_t = pm.Normal(name='i_z_t', sigma=i_sd_z, dims=['features', 'time'])
        # i_z = pm.Deterministic(name='i_z', var=pm.math.concatenate([i_z_0[:, None], i_z_t], axis=1).cumsum(axis=1)[:, 1:],
        #                        dims=['features', 'time'])
        i_z = pm.GaussianRandomWalk(name="i_z", init=pm.Normal.dist(mu=0, sigma=1, shape=F),
                                    mu=0, sigma=i_sd_z, dims=("time", "features"))

        # Emission distribution of the observed variable X
        i_x = pm.Normal(name="i_x", mu=i_z, sigma=i_sd_x, observed=x_obs)

        # pm.model_to_graphviz(hmm).view()

        idata = pm.sample(1000, init="adapt_diag", tune=1000, chains=2, random_seed=0, return_inferencedata=True)

        az.plot_trace(idata, var_names=["i_sd_z", "i_sd_x"])
        plt.show()

        m1 = idata.posterior["i_z"].sel(features=0).mean(dim=["chain", "draw"])
        sd1 = idata.posterior["i_z"].sel(features=0).std(dim=["chain", "draw"])
        # m1 = idata.posterior["i_z"].mean(dim=["chain", "draw"])
        # sd1 = idata.posterior["i_z"].std(dim=["chain", "draw"])

        plt.figure(figsize=(15, 8))
        plt.plot(range(T), z[:, 0], label="Real", color="tab:blue", marker="o")
        plt.plot(range(T), m1, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), m1 - sd1, m1 + sd1, color="tab:orange", alpha=0.4)
        plt.legend()
        plt.show()


        # mean_c = idata.posterior["i_z"].sel()
        # std_c = trace["i_uc"][::5].std(axis=0)[:, 0, 0]
        # plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        # plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        # plt.show()

        #
        # pm.set_data(
        #     {
        #         "x_obs": xs
        #     },
        #     coords={"idx": np.arange(N)},
        # )

        # fig = plt.figure(figsize=(15, 8))
        # mean_c = idata.posterior["i_z"].sel()
        # std_c = trace["i_uc"][::5].std(axis=0)[:, 0, 0]
        # plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        # plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        # plt.show()
