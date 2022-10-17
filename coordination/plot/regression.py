import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge


def plot_coordination_vs_score_regression(coordination: np.ndarray, scores: np.ndarray, regressor: BayesianRidge,
                                          line_label: str) -> plt.figure:
    fig = plt.figure(figsize=(6, 4))

    # Coordination points
    plt.scatter(coordination, scores, c="tab:blue", s=20, marker="x")

    # Line
    xs_line = np.linspace(0, 1, 100)[:, np.newaxis]
    ys_line, ys_std_line = regressor.predict(xs_line, return_std=True)
    plt.plot(xs_line, ys_line, color="tab:red", label=line_label, linewidth="0.5")
    plt.fill_between(xs_line.flatten(), ys_line - ys_std_line, ys_line + ys_std_line, color='tab:red', alpha=0.2)

    plt.xlabel(f"Average Coordination")
    plt.ylabel("Final Team Score")
    plt.xlim([0, 1])
    plt.ylim([0, 950])
    plt.title("Coordination vs Team Score")
    plt.legend()

    return fig
