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

    def __init__(self, tau=None, init=pm.Flat.dist(), sigma=None, mu=0.0, sd=None, c=None, *args, **kwargs):
        super().__init__(tau=tau, init=init, sigma=sigma, mu=mu, sd=sd, *args, **kwargs)

        # Coordination
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
        mu, sigma = self._mu_and_sigma(self.mu, self.sigma)

        p1 = pm.Mixture.dist(w=np.array([0.5, 0.5]), comp_dists=[
            pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 0] + mu, sigma=sigma, shape=(T - 1, C)),
            pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 0] + mu, sigma=sigma, shape=(T - 1, C)),
        ], shape=(T - 1, C)).logp(x_i[:, 0])

        p2 = pm.Mixture.dist(w=np.array([0.5, 0.5]), comp_dists=[
            pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 1] + mu, sigma=sigma, shape=(T - 1, C)),
            pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 1] + mu, sigma=sigma, shape=(T - 1, C)),
        ], shape=(T - 1, C)).logp(x_i[:, 1])

        p3 = pm.Mixture.dist(w=np.array([0.5, 0.5]), comp_dists=[
            pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 2] + mu, sigma=sigma, shape=(T - 1, C)),
            pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 2] + mu, sigma=sigma, shape=(T - 1, C)),
        ], shape=(T - 1, C)).logp(x_i[:, 2])

        return tt.sum(self.init.logp(x[0])) + tt.sum(p1) + tt.sum(p2) + tt.sum(p3)


if __name__ == "__main__":
    T = 50
    N = 2
    C = 10

    suc = 0.1
    vuc = np.square(suc)
    sc = 0.1
    vc = np.square(sc)
    sa = 1
    saa = 1
    so = 1
    w = np.ones(2) * 1 / 2

    uc = np.zeros((T, N, 1))
    c = np.zeros((T, N, 1))

    b = np.zeros((T, N, 3, C))
    o = np.zeros((T, N, 3, C))

    np.random.seed(1)
    for t in range(T):
        if t == 0:
            uc[0] = logit(0.5)
            # uc[0] = 0.5
        else:
            uc[t] = norm(loc=uc[t - 1], scale=suc).rvs()

        clipped_uc = np.clip(sigmoid(uc[t]), 1e-6, 1 - 1e-6)
        clipped_vc = np.minimum(vc, 2 * clipped_uc * (1 - clipped_uc))
        c[t] = beta(clipped_uc, clipped_vc).rvs()
        # c[t] = norm(loc=sigmoid(uc[t]), scale=sc).rvs()

        if t == 0:
            b[0] = norm(loc=0, scale=sa).rvs(size=(N, 3, C))
        else:
            u = np.random.rand()
            if u <= w[0]:
                b[t, :, 0] = norm(loc=b[t - 1, :, 1] * c[t] + b[t - 1, :, 0] * (1 - c[t]), scale=saa).rvs()
            else:
                b[t, :, 0] = norm(loc=b[t - 1, :, 2] * c[t] + b[t - 1, :, 0] * (1 - c[t]), scale=saa).rvs()

            u = np.random.rand()
            if u <= w[0]:
                b[t, :, 1] = norm(loc=b[t - 1, :, 0] * c[t] + b[t - 1, :, 1] * (1 - c[t]), scale=saa).rvs()
            else:
                b[t, :, 1] = norm(loc=b[t - 1, :, 2] * c[t] + b[t - 1, :, 1] * (1 - c[t]), scale=saa).rvs()

            u = np.random.rand()
            if u <= w[0]:
                b[t, :, 2] = norm(loc=b[t - 1, :, 0] * c[t] + b[t - 1, :, 2] * (1 - c[t]), scale=saa).rvs()
            else:
                b[t, :, 2] = norm(loc=b[t - 1, :, 1] * c[t] + b[t - 1, :, 2] * (1 - c[t]), scale=saa).rvs()

        o[t] = norm(loc=b[t], scale=so).rvs()

    # Model in PyMC3
    with pm.Model() as model:
        # data = pm.Data("data", c.T)
        # obs1 = pm.Data("obs1", o1, dims="time", export_index_as_coords=True)
        # obs2 = pm.Data("obs2", o2, dims="time", export_index_as_coords=True)
        # obs3 = pm.Data("obs3", o3, dims="time", export_index_as_coords=True)
        # obs_o = pm.Data("obs", o, shape=())
        # obs_b = pm.Data("obs_b", b)
        # obs_c = pm.Data("obs_c", c)
        # obs_uc = pm.Data("obs_uc", uc)

        # i_vuc = pm.InverseGamma(name="i_vuc", alpha=1e-6, beta=1e-6)
        # i_vc = pm.Uniform(name="i_vc", alpha=1e-6, beta=1e-6, testval=0.01)
        # i_vc = pm.Flat(name="i_vc")
        # i_vaa = pm.InverseGamma(name="i_vaa", alpha=1, beta=1)
        # i_vo = pm.InverseGamma(name="i_vo", alpha=1, beta=1)
        i_suc = pm.HalfNormal(name="i_suc", sigma=1, observed=None)
        i_sc = pm.HalfNormal(name="i_sc", sigma=0.1, observed=sc, testval=1e-6)
        i_saa = pm.HalfNormal(name="i_saa", sigma=1, observed=None)
        i_so = pm.HalfNormal(name="i_so", sigma=1, observed=None)

        # i_vaa_print = tt.printing.Print("i_vaa")(i_vaa)
        # i_vo_print = tt.printing.Print("i_vo")(i_vo)

        # i_suc = pm.Deterministic("i_suc", pm.math.sqrt(i_vuc))
        # i_sc = pm.Deterministic("i_sc", pm.math.sqrt(i_vc))
        # i_saa = pm.Deterministic("i_saa", pm.math.sqrt(i_vaa_print))
        # i_so = pm.Deterministic("i_so", pm.math.sqrt(i_vo_print))

        for i in range(N):
            i_uc = pm.GaussianRandomWalk(name=f"i_uc_{i}", sigma=i_suc, shape=(T, 1), testval=logit(0.5), observed=None)

            # i_c = pm.Deterministic("i_mc", pm.math.sigmoid(i_uc))

            i_mc = pm.Deterministic(f"i_mc_{i}", pm.math.sigmoid(i_uc))
            i_mc_clipped = pm.Deterministic(f"i_mc_clipped_{i}", pm.math.clip(i_mc, 1e-6, 1 - 1e-6))
            # # i_c = pm.Normal(name="i_c", mu=i_mc, sigma=sc, shape=(T, N, 1), observed=c)

            i_clipped_sigma = pm.Deterministic(f"i_clipped_sigma_{i}",
                                               pm.math.minimum(i_sc, 2 * i_mc_clipped * (1 - i_mc_clipped)))
            i_c = pm.Beta(name=f"i_c_{i}", mu=i_mc_clipped, sigma=i_clipped_sigma, shape=(T, 1), observed=None)

            latent_prior = pm.Normal.dist(mu=0, sigma=sa, shape=(3, C))
            i_b = LatentComponentRandomWalk(name=f"i_b_{i}", init=latent_prior, sigma=i_saa, shape=(T, 3, C), c=i_c,
                                            observed=None)

            i_o = pm.Normal(name=f"i_o_{i}", mu=i_b, sigma=i_so, shape=(T, 3, C), observed=o[:, i])

        theano.config.floatX = 'float64'
        trace = pm.sample(500, init="adapt_diag", return_inferencedata=False, tune=500, chains=2, random_seed=0)

        az.plot_trace(trace, var_names=[
            "i_suc",
            # "i_sc",
            "i_saa",
            "i_so"])
        plt.show()

        fig = plt.figure(figsize=(15, 8))
        plt.plot(range(T), c[:, 0, 0], label="Real", color="tab:blue", marker="o")
        #
        # mean_c = sigmoid(trace["i_uc"][::5]).mean(axis=0)[:, 0, 0]
        # std_c = sigmoid(trace["i_uc"][::5]).std(axis=0)[:, 0, 0]
        mean_c = trace["i_c_0"][::5].mean(axis=0)[:, 0]
        std_c = trace["i_c_0"][::5].std(axis=0)[:, 0]
        plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        plt.show()

        fig = plt.figure(figsize=(15, 8))
        plt.plot(range(T), c[:, 1, 0], label="Real", color="tab:blue", marker="o")
        #
        # mean_c = sigmoid(trace["i_uc"][::5]).mean(axis=0)[:, 0, 0]
        # std_c = sigmoid(trace["i_uc"][::5]).std(axis=0)[:, 0, 0]
        mean_c = trace["i_c_1"][::5].mean(axis=0)[:, 0]
        std_c = trace["i_c_1"][::5].std(axis=0)[:, 0]
        plt.plot(range(T), mean_c, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), mean_c - std_c, mean_c + std_c, color="tab:orange", alpha=0.4)
        plt.show()
