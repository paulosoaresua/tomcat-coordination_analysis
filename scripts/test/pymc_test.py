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


def generate_data_fixed_influence():
    uc = np.zeros((N, T))
    c = np.zeros((N, T))

    b = np.zeros((N, T, S, C))
    o = np.zeros((N, T, S, C))

    np.random.seed(1)
    VC = np.square(SC)

    u1 = np.random.rand(N)
    u2 = np.random.rand(N)
    u3 = np.random.rand(N)

    for t in range(T):
        if t == 0:
            uc[:, 0] = logit(0.5)
        else:
            uc[:, t] = norm(loc=uc[:, t - 1], scale=SUC).rvs()

        clipped_uc = np.clip(sigmoid(uc[:, t]), MIN_C, MAX_C)
        clipped_vc = np.minimum(VC, clipped_uc * (1 - clipped_uc) / 2)
        c[:, t] = beta(clipped_uc, clipped_vc).rvs()

        if t == 0:
            b[:, 0] = norm(loc=0, scale=SA).rvs(size=(N, S, C))
        else:
            b[:, t, 0] = np.where(u1[:, None] <= W[0],
                                  norm(loc=b[:, t - 1, 1] * c[:, t][:, None] + b[:, t - 1, 0] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs(),
                                  norm(loc=b[:, t - 1, 2] * c[:, t][:, None] + b[:, t - 1, 0] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs())

            b[:, t, 1] = np.where(u2[:, None] <= W[0],
                                  norm(loc=b[:, t - 1, 0] * c[:, t][:, None] + b[:, t - 1, 1] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs(),
                                  norm(loc=b[:, t - 1, 2] * c[:, t][:, None] + b[:, t - 1, 1] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs())

            b[:, t, 2] = np.where(u3[:, None] <= W[0],
                                  norm(loc=b[:, t - 1, 0] * c[:, t][:, None] + b[:, t - 1, 2] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs(),
                                  norm(loc=b[:, t - 1, 1] * c[:, t][:, None] + b[:, t - 1, 2] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs())

            o[:, t] = norm(loc=b[:, t], scale=SO).rvs()

    return uc, c, b, o


def generate_data():
    uc = np.zeros((N, T))
    c = np.zeros((N, T))

    b = np.zeros((N, T, S, C))
    o = np.zeros((N, T, S, C))

    np.random.seed(1)
    VC = np.square(SC)
    for t in range(T):
        if t == 0:
            uc[:, 0] = logit(0.5)
        else:
            uc[:, t] = norm(loc=uc[:, t - 1], scale=SUC).rvs()

        clipped_uc = np.clip(sigmoid(uc[:, t]), MIN_C, MAX_C)
        clipped_vc = np.minimum(VC, clipped_uc * (1 - clipped_uc) / 2)
        c[:, t] = beta(clipped_uc, clipped_vc).rvs()

        # Influence changes at every time step
        u1 = np.random.rand(N)
        u2 = np.random.rand(N)
        u3 = np.random.rand(N)
        if t == 0:
            b[:, 0] = norm(loc=0, scale=SA).rvs(size=(N, S, C))
        else:
            b[:, t, 0] = np.where(u1[:, None] <= W[0],
                                  norm(loc=b[:, t - 1, 1] * c[:, t][:, None] + b[:, t - 1, 0] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs(),
                                  norm(loc=b[:, t - 1, 2] * c[:, t][:, None] + b[:, t - 1, 0] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs())

            b[:, t, 1] = np.where(u2[:, None] <= W[0],
                                  norm(loc=b[:, t - 1, 0] * c[:, t][:, None] + b[:, t - 1, 1] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs(),
                                  norm(loc=b[:, t - 1, 2] * c[:, t][:, None] + b[:, t - 1, 1] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs())

            b[:, t, 2] = np.where(u3[:, None] <= W[0],
                                  norm(loc=b[:, t - 1, 0] * c[:, t][:, None] + b[:, t - 1, 2] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs(),
                                  norm(loc=b[:, t - 1, 1] * c[:, t][:, None] + b[:, t - 1, 2] * (1 - c[:, t][:, None]),
                                       scale=SAA).rvs())

            o[:, t] = norm(loc=b[:, t], scale=SO).rvs()

    return uc, c, b, o


def mixture_component_fixed_influence_logp(b: at.TensorVariable, c: at.TensorVariable, sigma: at.TensorVariable):
    x_im1 = b[:, :, :, :-1]
    x_i = b[:, :, :, 1:]
    c_i = c[:, 1:][:, None, :]
    # c_i = c[:, 1:][:, None, None, :]

    num_trials, num_subjects, num_channels, num_time_steps = b.shape
    shape = (num_trials, num_channels, num_time_steps - 1)
    # shape = (num_trials, num_subjects, num_channels, num_time_steps - 1)

    sum = 0
    for s1 in range(S):
        w = 0
        sum_tmp = 0
        for s2 in range(S):
            if s1 == s2:
                continue

            logp_s1_from_s2 = pm.logp(
                pm.Normal.dist(mu=x_im1[:, s2] * c_i + (1 - c_i) * x_im1[:, s1], sigma=sigma, shape=shape),
                x_i[:, s1])

            sum_tmp += W[w] * pm.math.exp(logp_s1_from_s2)

        sum += pm.math.log(sum_tmp).sum()


    #
    # p1 = pm.math.log(W[0] * pm.math.exp(
    #     pm.logp(
    #         pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 0], sigma=sigma, shape=shape),
    #         x_i[:, 0])) + W[1] * pm.math.exp(pm.logp(
    #     pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 0], sigma=sigma, shape=shape),
    #     x_i[:, 0])))
    #
    # p2 = pm.math.log(
    #     W[0] * pm.math.exp(
    #         pm.logp(pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 1], sigma=sigma, shape=shape),
    #                 x_i[:, 1])) + W[1] * pm.math.exp(pm.logp(
    #         pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 1], sigma=sigma, shape=shape), x_i[:, 1])))
    #
    # p3 = pm.math.log(W[0] * pm.math.exp(
    #     pm.logp(pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 2], sigma=sigma, shape=shape),
    #             x_i[:, 2])) + W[1] * pm.math.exp(pm.logp(
    #     pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 2], sigma=sigma, shape=shape), x_i[:, 2])))

    init_dist = pm.Normal.dist(mu=0, sigma=1, shape=(
        num_trials, num_subjects, num_channels))

    return at.sum(pm.logp(init_dist, b[:, :, :, 0])) + sum

    # return at.sum(pm.logp(init_dist, b[:, :, :, 0])) + at.sum(p1) + at.sum(p2) + at.sum(p3)


def mixture_component_logp(b: at.TensorVariable, c: at.TensorVariable, w: at.TensorVariable, sigma: at.TensorVariable):
    x_im1 = b[:, :, :, :-1]
    x_i = b[:, :, :, 1:]
    c_i = c[:, 1:][:, None, :]
    w_i = w[:, :, :, 1:]

    num_trials, num_subjects, num_channels, num_time_steps = b.shape
    shape = (num_trials, num_channels, num_time_steps - 1)
    # shape = (num_trials, num_subjects, num_channels, num_time_steps - 1)

    p1 = pm.math.log(w_i[:, 0, 0][:, None, :] * pm.math.exp(
        pm.logp(pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 0], sigma=sigma, shape=shape),
                x_i[:, 0])) + w_i[:, 0, 1][:, None, :] * pm.math.exp(pm.logp(
        pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 0], sigma=sigma, shape=shape),
        x_i[:, 0])))

    p2 = pm.math.log(
        w_i[:, 1, 0][:, None, :] * pm.math.exp(
            pm.logp(pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 1], sigma=sigma, shape=shape),
                    x_i[:, 1])) + w_i[:, 1, 1][:, None, :] * pm.math.exp(pm.logp(
            pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 1], sigma=sigma, shape=shape), x_i[:, 1])))

    p3 = pm.math.log(w_i[:, 2, 0][:, None, :] * pm.math.exp(
        pm.logp(pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 2], sigma=sigma, shape=shape),
                x_i[:, 2])) + w_i[:, 2, 1][:, None, :] * pm.math.exp(pm.logp(
        pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 2], sigma=sigma, shape=shape), x_i[:, 2])))

    init_dist = pm.Normal.dist(mu=0, sigma=1, shape=(
        num_trials, num_subjects, num_channels))

    return at.sum(pm.logp(init_dist, b[:, :, :, 0])) + at.sum(p1) + at.sum(p2) + at.sum(p3)


if __name__ == "__main__":
    # import warnings
    #
    # warnings.filterwarnings("ignore")

    uc, c, b, o = generate_data()

    coords = {"trial": np.arange(N), "subject": np.arange(S), "channel": np.arange(C), "time": np.arange(T), "body_feature": np.arange(1),}
    with pm.Model(coords=coords) as model:
        obs = pm.MutableData("obs", np.swapaxes(np.swapaxes(o, 1, 2), 2, -1),
                             dims=("trial", "subject", "channel", "time"))
        obs_b = np.swapaxes(np.swapaxes(b, 1, 2), 2, -1)

        i_suc = pm.HalfNormal(name="i_suc", sigma=1, observed=None)
        i_saa = pm.HalfNormal(name="i_saa", sigma=1, observed=None)
        i_so = pm.HalfNormal(name="i_so", sigma=1, observed=None)

        i_uc = pm.GaussianRandomWalk(name=f"i_uc", init_dist=pm.Normal.dist(mu=0, sigma=1, shape=N),
                                     sigma=i_suc, dims=["trial", "time"], initval=logit(np.ones((N, T)) * 0.5),
                                     observed=None)

        i_mc = pm.Deterministic(f"i_mc", pm.math.sigmoid(i_uc), dims=["trial", "time"])
        i_mc_clipped = pm.Deterministic(f"i_mc_clipped", pm.math.clip(i_mc, MIN_C, MAX_C), dims=["trial", "time"])

        i_clipped_sigma = pm.Deterministic(f"i_clipped_sigma",
                                           pm.math.minimum(SC, 2 * i_mc_clipped * (1 - i_mc_clipped)))
        i_c = pm.Beta(name=f"i_c", mu=i_mc_clipped, sigma=i_clipped_sigma, dims=["trial", "time"], observed=None)
        # i_c = pm.Beta(name=f"i_c", mu=i_mc, sigma=SC, dims=["trial", "time"], observed=None)

        i_b = pm.DensityDist("i_b", at.as_tensor_variable(i_c), at.as_tensor_variable(i_saa),
                             logp=mixture_component_fixed_influence_logp, dims=["trial", "subject", "channel", "time"],
                             observed=None)

        i_o = pm.Normal(name=f"i_o", mu=i_b, sigma=i_so, observed=obs)

        idata = pm.sample(1000, init="adapt_diag", tune=1000, chains=2, random_seed=0)

        az.plot_trace(idata, var_names=["i_suc", "i_saa", "i_so"])
        plt.show()

        # m1 = idata.posterior["i_uc"].sel(trial=0).mean(dim=["chain", "draw"])
        # sd1 = idata.posterior["i_uc"].sel(trial=0).std(dim=["chain", "draw"])
        #
        # plt.figure(figsize=(15, 8))
        # plt.plot(range(T), uc[0], label="Real", color="tab:blue", marker="o")
        # plt.plot(range(T), m1, label="Inferred", color="tab:orange", marker="o")
        # plt.fill_between(range(T), m1 - sd1, m1 + sd1, color="tab:orange", alpha=0.4)
        # plt.title("Unbounded Coordination")
        # plt.legend()
        # plt.show()

        m1 = idata.posterior["i_c"].sel(trial=0).mean(dim=["chain", "draw"])
        sd1 = idata.posterior["i_c"].sel(trial=0).std(dim=["chain", "draw"])

        plt.figure(figsize=(15, 8))
        plt.plot(range(T), c[0], label="Real", color="tab:blue", marker="o")
        plt.plot(range(T), m1, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(T), m1 - sd1, m1 + sd1, color="tab:orange", alpha=0.4)
        plt.title("Coordination")
        plt.legend()
        plt.show()
