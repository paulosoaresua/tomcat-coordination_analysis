"""
Toy script to infer the mean and variance of an emission distribution using Gaussian-IG distribution.

The model is consisted of a 2D latent variable X with prior N(0,1) and an observation Y such that Y ~ N(Xm, v), where
m, v ~ N-IG(mu, nu, a, b). We don't need to implement a sampler for N-IG because we always use the mean of the posterior
during training. In this case, E[m] = mu* and E[v] = b*/(a* - 1) where the star parameters are computed by the
sufficient statistics of the samples generated in each Gibbs Sampling step.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

NUM_SAMPLES = 10000
TRUE_M = np.array([30, 200])
TRUE_V = np.array([5, 10])
MU = 0
NU = 1E-6
A = 1E-6
B = 1E-6

# X = np.ones((NUM_SAMPLES, 2))
X = norm(loc=np.zeros((NUM_SAMPLES, 2)), scale=1).rvs()
Y = norm(loc=X * TRUE_M, scale=np.sqrt(TRUE_V)).rvs()

print(np.var(Y, axis=0))

NUM_ITER = 100
# Parameter values:
ms = [np.array([1, 1])]
vs = [np.array([1, 1])]
for s in range(NUM_ITER):
    # Posterior of X
    aux = vs[-1] + ms[-1]
    # X_samples = np.ones_like(Y)
    X_samples = norm(loc=(ms[-1] * Y) / aux, scale=np.sqrt(1 / aux)).rvs()

    # Update parameters
    sum_xx = np.sum(X * X, axis=0)
    sum_xy = np.sum(X * Y, axis=0)
    sum_yy = np.sum(Y * Y, axis=0)

    m_star = (NU * MU + sum_xy) / (NU + NUM_SAMPLES)
    a_star = A + NUM_SAMPLES / 2
    b_star = B + (1 / 2) * ((sum_yy - (sum_xy ** 2) / sum_xx) + (NU * sum_xx / (NU + sum_xx)) * (
                MU - sum_xy / sum_xx) ** 2)

    ms.append(m_star)
    vs.append(np.maximum(b_star / (a_star - 1), 1e-3))

print(ms[-1])
print(vs[-1])

plt.figure()
plt.plot(range(len(ms)), np.array(ms)[:, 0], label="Mean")
plt.plot(range(len(vs)), np.array(vs)[:, 0], label="Var")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(ms)), np.array(ms)[:, 1], label="Mean")
plt.plot(range(len(vs)), np.array(vs)[:, 1], label="Var")
plt.legend()
plt.show()
