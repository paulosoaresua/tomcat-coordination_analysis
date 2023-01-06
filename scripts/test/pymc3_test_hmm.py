import numpy as np
from scipy.stats import norm

import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano

from coordination.common.distribution import beta
from coordination.common.utils import logit, sigmoid


if __name__ == "__main__":
    T = 50
    N = 1
    C = 1

    sc = 0.1
    so = 1

    c = np.zeros((T, N, C))
    o = np.zeros((T, N, 3, C))

    np.random.seed(1)
    for t in range(T):
        if t == 0:
            c[0] = 0.5
        else:
            c[t] = norm(loc=c[t - 1], scale=sc).rvs()

        o[t] = norm(loc=c[t], scale=so).rvs()

    hmm = pm.Model()
    # Model in PyMC3
    with hmm:
        i_sc = pm.HalfNormal(name="i_sc", sigma=1, observed=suc)
        i_so = pm.HalfNormal(name="i_so", sigma=1, observed=so)

        # i_vaa_print = tt.printing.Print("i_vaa")(i_vaa)
        # i_vo_print = tt.printing.Print("i_vo")(i_vo)

        # i_suc = pm.Deterministic("i_suc", pm.math.sqrt(i_vuc))
        # i_sc = pm.Deterministic("i_sc", pm.math.sqrt(i_vc))
        # i_saa = pm.Deterministic("i_saa", pm.math.sqrt(i_vaa_print))
        # i_so = pm.Deterministic("i_so", pm.math.sqrt(i_vo_print))

        i_uc = pm.GaussianRandomWalk(name="i_uc", sigma=i_suc, shape=(T, N, 1), observed=None)

        i_c = i_uc #pm.Deterministic("i_mc", pm.math.sigmoid(i_uc))

        # i_mc = pm.Deterministic("i_mc", pm.math.sigmoid(i_uc))
        # i_mc_clipped = pm.Deterministic("i_mc_clipped", pm.math.clip(i_mc, 1e-4, 1 - 1e-4))
        # # i_c = pm.Normal(name="i_c", mu=i_mc, sigma=sc, shape=(T, N, 1), observed=c)
        # i_c = pm.Beta(name="i_c", mu=i_mc_clipped, sigma=i_sc, shape=(T, N, 1), observed=c)

        latent_prior = pm.Normal.dist(mu=0, sigma=sa, shape=(N, 3, C))
        i_b = LatentComponentRandomWalk(name="i_b", init=latent_prior, sigma=i_saa, shape=(T, N, 3, C), c=i_c,
                                        observed=b)

        i_o = pm.Normal(name="i_o", mu=i_b, sigma=i_so, shape=(T, N, 3, C), observed=o)

        theano.config.floatX = 'float64'
        trace = pm.sample(1000, init="adapt_diag", return_inferencedata=False, tune=500, chains=2, random_seed=0)
        # ,
        #               initvals={"i_vuc": 1, "i_vc": 1e-4, "i_vaa": 1, "i_vo": 1, "i_c": np.ones((T, N, 1)) * 0.5,
        #                         "i_uc": np.ones((T, N, 1)) * logit(0.5)})

        # az.plot_trace(trace, var_names=["i_saa", "i_so"])
        # plt.show()

        fig = plt.figure(figsize=(15, 8))
        # plt.plot(range(T), c[:, 0, 0], label="Real", color="tab:blue", marker="o")
        #
        # # mean_c = sigmoid(trace["i_uc"][::5]).mean(axis=0)[:, 0, 0]
        # # std_c = sigmoid(trace["i_uc"][::5]).std(axis=0)[:, 0, 0]
        mean_c = trace["i_uc"][::5].mean(axis=0)[:, 0, 0]
        std_c = trace["i_uc"][::5].std(axis=0)[:, 0, 0]
        plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        plt.show()
