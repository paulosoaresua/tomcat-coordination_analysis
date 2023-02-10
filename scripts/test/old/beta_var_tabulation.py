import numpy as np
from coordination.common.distribution import beta

import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    M = np.linspace(0.001, 0.999, 100)
    X = np.linspace(0.001, 0.999, 100)
    S = np.linspace(0.001, 0.999, 1000)

    MM, XX = np.meshgrid(X, M)

    Z = np.ones((1000, 100, 100)) * -np.inf

    for i, s in tqdm(enumerate(S)):
        Z[i] = np.where(MM * (1 - MM) < s, -np.inf, beta(MM, s).logpdf(XX))

    Z2 = S[np.argmax(Z, axis=0)]

    fig = plt.figure(figsize=(10, 14))
    ax = fig.gca(projection="3d")
    ax.plot_surface(MM, XX, Z2)
    plt.show()

    print(M[50])
    print(X[50])
    print(Z2[50, 50])

    # M = np.random.rand(3)
    # X = beta(M, 0.05)
