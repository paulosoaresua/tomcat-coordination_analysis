
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(1234)

    N = 10
    SCALE = 0.25

    def true_function(x: np.ndarray):
        # return 4 + 5 * x
        return np.sin(2 * np.pi * x)

    def generate_noisy_data(x: np.ndarray, scale: float):
        y = true_function(x) + np.random.normal(scale=scale, size=len(x))
        return y

    def transform_features(X, m):
        """ Create a polynomial of specified degrees """
        return PolynomialFeatures(degree=m).fit_transform(X.reshape(-1, 1))

    x_plot = np.arange(0, 1.01, 0.01)
    y_plot = true_function(x_plot)

    x_train = np.linspace(0, 1, N)
    y_train = generate_noisy_data(x_train, SCALE)

    x_test = np.arange(0, 1.01, 0.01)
    y_test = generate_noisy_data(x_test, SCALE)

    M = 10
    features_train = transform_features(x_train, m=M-1)
    features_test = transform_features(x_test, m=M-1)

    # alpha = 5e-3
    # beta = (1/SCALE) ** 2

    # Find alpha and beta
    max_iter = 10000
    data_cov = features_train.T @ features_train
    eig_hat = np.linalg.eig(data_cov)[0]
    alpha = 1E-5
    beta = 20

    # alpha = 5e-3
    # alpha = 1
    # beta = (1 / SCALE) ** 2

    num_iter = 0
    converged = False
    alphas = [alpha]
    betas = [beta]
    while not converged and num_iter < max_iter:
        # Re-estimate posterior
        S_inv = alpha * np.eye(M, M) + beta * data_cov
        S = np.linalg.inv(S_inv)
        mean = beta * S @ features_train.T @ y_train
        eig = beta * eig_hat
        gamma = np.sum(eig / (alpha + eig))

        # Re-estimate alpha and beta
        new_alpha = gamma / (mean.T @ mean)
        beta_inv = np.sum(np.square(y_train - features_train @ mean.T)) / (N - gamma)
        new_beta = 1 / beta_inv

        num_iter += 1
        # converged = np.abs(new_alpha - alpha) < 1E-10 and np.abs(new_beta - beta) < 1E-10
        # converged = np.abs(new_alpha - alpha) < 1E-10
        converged = np.abs(new_beta - beta) < 1E-10
        alpha = new_alpha
        beta = new_beta

        alphas.append(alpha)
        betas.append(beta)

    print(num_iter)
    print(alpha)
    print(beta)

    # alpha = 5e-3
    # beta = (1 / SCALE) ** 2

    # Posterior covariance matrix
    S_inv = alpha * np.eye(M, M) + beta * data_cov
    S = np.linalg.inv(S_inv)

    # Posterior mean
    mean = beta * S @ features_train.T @ y_train

    # Predictive distribution
    y_pred_mean = features_test @ mean
    y_pred_var = 1/beta + np.sum(features_test @ S * features_test, axis=1)

    # Plot predictions
    plt.figure()
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_plot, y_plot, c="g", label="$4 + 5x$")
    plt.plot(x_test, y_pred_mean, c="r", label="mean")
    plt.fill_between(x_test, y_pred_mean - np.sqrt(y_pred_var), y_pred_mean + np.sqrt(y_pred_var), color="pink",
                     label="std.", alpha=0.5)
    plt.xlim(-0.1, 1.1)
    # plt.ylim(-1.5, 1.5)
    plt.annotate(f"M={M-1}", xy=(0.8, 1))
    plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0)
    plt.show()

    # Plot alphas
    plt.figure()
    plt.plot(np.arange(len(alphas)), alphas, label="alpha")
    plt.legend()
    plt.show()

    # Plot betas
    plt.figure()
    plt.plot(np.arange(len(betas)), betas, label="beta")
    plt.legend()
    plt.show()



