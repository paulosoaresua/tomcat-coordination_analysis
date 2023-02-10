import numpy as np
from scipy.stats import norm

import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import theano.tensor as tt
import theano

from coordination.common.distribution import beta
from coordination.common.utils import logit, sigmoid


class LatentComponentRandomWalk(pm.GaussianRandomWalk):

    # def __new__(cls, init=pm.Flat.dist(), sigma=None, mu=0.0, c=None, *args, **kwargs):
    #     self = super().__new__(cls, init=init, sigma=sigma, mu=mu, *args, **kwargs)
    #
    #     # Coordination
    #     self.c = tt.as_tensor_variable(c)
    #     return self

    def __init__(self, init=pm.Flat.dist(), sigma=None, mu=0.0, c=None, *args, **kwargs):
        super().__init__(init=init, sigma=sigma, mu=mu, *args, **kwargs)
        # Coordination
        self.init = init
        self.mu = tt.as_tensor_variable(mu)
        self.sigma = tt.as_tensor_variable(sigma)
        self.c = tt.as_tensor_variable(c)

    def logp(self, x):
        """
        Calculate log-probability of Gaussian Random Walk distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        x_im1 = x[:-1]
        x_i = x[1:]
        c_i = self.c[1:]

        if self.mu.ndim > 0:
            mu = self.mu[1:]
        else:
            mu = self.mu

        if self.sigma.ndim > 0:
            sigma = self.sigma[1:]
        else:
            sigma = self.sigma

        p1 = 0.5 * pm.Normal.dist(mu=x_im1[:, :, 1] * c_i[:, :, None] + (1 - c_i[:, :, None]) * x_im1[:, :, 0] + mu,
                                  sigma=sigma, shape=(T - 1, N, C)).logp(x_i[:, :, 0]) + \
             0.5 * pm.Normal.dist(mu=x_im1[:, :, 2] * c_i[:, :, None] + (1 - c_i[:, :, None]) * x_im1[:, :, 0] + mu,
                                  sigma=sigma, shape=(T - 1, N, C)).logp(x_i[:, :, 0])

        p2 = 0.5 * pm.Normal.dist(mu=x_im1[:, :, 0] * c_i[:, :, None] + (1 - c_i[:, :, None]) * x_im1[:, :, 1] + mu,
                                  sigma=sigma, shape=(T - 1, N, C)).logp(x_i[:, :, 1]) + \
             0.5 * pm.Normal.dist(mu=x_im1[:, :, 2] * c_i[:, :, None] + (1 - c_i[:, :, None]) * x_im1[:, :, 1] + mu,
                                  sigma=sigma, shape=(T - 1, N, C)).logp(x_i[:, :, 1])

        p3 = 0.5 * pm.Normal.dist(mu=x_im1[:, :, 0] * c_i[:, :, None] + (1 - c_i[:, :, None]) * x_im1[:, :, 2] + mu,
                                  sigma=sigma, shape=(T - 1, N, C)).logp(x_i[:, :, 2]) + \
             0.5 * pm.Normal.dist(mu=x_im1[:, :, 1] * c_i[:, :, None] + (1 - c_i[:, :, None]) * x_im1[:, :, 2] + mu,
                                  sigma=sigma, shape=(T - 1, N, C)).logp(x_i[:, :, 2])

        return tt.sum(self.init.logp(x[0])) + tt.sum(p1) + tt.sum(p2) + tt.sum(p3)


if __name__ == "__main__":
    T = 50
    N = 2
    C = 1

    suc = 0.1
    vuc = np.square(suc)
    sc = 0.1
    vc = np.square(sc)
    sa = 1
    saa = 1
    so = 1
    w = np.ones(2) * 1 / 2

    uc = np.zeros((T, N))
    c = np.zeros((T, N))

    b = np.zeros((T, N, 3, C))
    o = np.zeros((T, N, 3, C))

    np.random.seed(1)

    u1 = np.random.rand(N)
    u2 = np.random.rand(N)
    u3 = np.random.rand(N)
    for t in range(T):
        if t == 0:
            uc[0] = np.ones(N) * logit(0.5)
            # uc[0] = 0.5
        else:
            uc[t] = norm(loc=uc[t - 1], scale=suc).rvs()

        clipped_uc = np.clip(sigmoid(uc[t]), 1e-6, 1 - 1e-6)
        clipped_vc = np.minimum(vc, clipped_uc * (1 - clipped_uc) / 2)
        c[t] = beta(clipped_uc, clipped_vc).rvs()

        if t == 0:
            b[0] = norm(loc=0, scale=sa).rvs(size=(N, 3, C))
        else:
            b[t, :, 0] = np.where(u1[:, None] <= .5,
                                  norm(loc=b[t - 1, :, 1] * c[t][:, None] + b[t - 1, :, 0] * (1 - c[t][:, None]),
                                       scale=saa).rvs(),
                                  norm(loc=b[t - 1, :, 2] * c[t][:, None] + b[t - 1, :, 0] * (1 - c[t][:, None]),
                                       scale=saa).rvs())

            b[t, :, 1] = np.where(u2[:, None] <= .5,
                                  norm(loc=b[t - 1, :, 0] * c[t][:, None] + b[t - 1, :, 1] * (1 - c[t][:, None]),
                                       scale=saa).rvs(),
                                  norm(loc=b[t - 1, :, 2] * c[t][:, None] + b[t - 1, :, 1] * (1 - c[t][:, None]),
                                       scale=saa).rvs())

            b[t, :, 2] = np.where(u3[:, None] <= .5,
                                  norm(loc=b[t - 1, :, 0] * c[t][:, None] + b[t - 1, :, 2] * (1 - c[t][:, None]),
                                       scale=saa).rvs(),
                                  norm(loc=b[t - 1, :, 1] * c[t][:, None] + b[t - 1, :, 2] * (1 - c[t][:, None]),
                                       scale=saa).rvs())

        o[t] = norm(loc=b[t], scale=so).rvs()

    # import warnings
    #
    # warnings.filterwarnings("ignore")

    # Model in PyMC3
    coords = {"subjects": np.arange(3), "channels": np.arange(C), "time": np.arange(T), "trials": np.arange(N)}
    with pm.Model(coords=coords) as model:
        data = pm.Data("data", o, dims=["time", "trials", "subjects", "channels"])

        i_suc = pm.HalfNormal(name="i_suc", sigma=1, observed=None)
        i_saa = pm.HalfNormal(name="i_saa", sigma=1, observed=None)
        i_so = pm.HalfNormal(name="i_so", sigma=1, observed=None)

        i_uc = pm.GaussianRandomWalk(name=f"i_uc", init=pm.Normal.dist(mu=0, sigma=1e-6, shape=N),
                                     sigma=i_suc, testval=np.ones((T, N)) * logit(0.5),
                                     dims=["time", "trials"], observed=None)

        i_mc = pm.Deterministic(f"i_mc", pm.math.sigmoid(i_uc), dims=["time", "trials"])
        i_mc_clipped = pm.Deterministic(f"i_mc_clipped", pm.math.clip(i_mc, 1e-6, 1 - 1e-6),
                                        dims=["time", "trials"])

        i_clipped_sigma = pm.Deterministic(f"i_clipped_sigma",
                                           pm.math.minimum(sc, 2 * i_mc_clipped * (1 - i_mc_clipped)),
                                           dims=["time", "trials"])
        i_c = pm.Beta(name=f"i_c", mu=i_mc_clipped, sigma=i_clipped_sigma, testval=np.ones((T, N)) * 0.5,
                      dims=["time", "trials"], observed=None)

        latent_prior = pm.Normal.dist(mu=0, sigma=sa, shape=(N, 3, C))
        i_b = LatentComponentRandomWalk(name=f"i_b", init=latent_prior, sigma=i_saa,
                                        dims=["time", "trials", "subjects", "channels"], c=i_c, observed=None)

        i_o = pm.Normal(name=f"i_o", mu=i_b, sigma=i_so, observed=data)

        print("Printing Model")
        pm.model_to_graphviz(model).view()

        print("Inference")
        idata = pm.sample(500, init="adapt_diag", tune=500, chains=2, random_seed=0, return_inferencedata=True)

        az.plot_trace(idata, var_names=[
            "i_suc",
            "i_saa",
            "i_so"])
        plt.show()

        m1 = idata.posterior["i_c"].sel(trials=0).mean(dim=["chain", "draw"])
        sd1 = idata.posterior["i_c"].sel(trials=0).std(dim=["chain", "draw"])

        plt.figure(figsize=(15, 8))
        plt.plot(range(T), c[:, 0], label="Real", color="tab:blue", marker="o")
        plt.plot(range(T), m1, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), m1 - sd1, m1 + sd1, color="tab:orange", alpha=0.4)
        plt.legend()
        plt.show()

        # az.plot_trace(idata, var_names=[
        #     "i_suc",
        #     "i_saa",
        #     "i_so"])
        # plt.show()
        #
        # plt.figure(figsize=(15, 8))
        # plt.plot(range(T), c[:, 0, 0], label="Real", color="tab:blue", marker="o")
        # #
        # # mean_c = sigmoid(trace["i_uc"][::5]).mean(axis=0)[:, 0, 0]
        # # std_c = sigmoid(trace["i_uc"][::5]).std(axis=0)[:, 0, 0]
        # mean_c = idata["i_c_0"][::5].mean(axis=0)[:, 0]
        # std_c = idata["i_c_0"][::5].std(axis=0)[:, 0]
        # plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        # plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        # plt.show()
        #
        # plt.figure(figsize=(15, 8))
        # plt.plot(range(T), c[:, 1, 0], label="Real", color="tab:blue", marker="o")
        # #
        # # mean_c = sigmoid(trace["i_uc"][::5]).mean(axis=0)[:, 0, 0]
        # # std_c = sigmoid(trace["i_uc"][::5]).std(axis=0)[:, 0, 0]
        # mean_c = idata["i_c_1"][::5].mean(axis=0)[:, 0]
        # std_c = idata["i_c_1"][::5].std(axis=0)[:, 0]
        # plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        # plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        # plt.show()
