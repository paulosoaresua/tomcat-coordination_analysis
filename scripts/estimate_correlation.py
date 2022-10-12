import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


def estimate(data_df_path: str, out_dir: str, plot_regression: bool):
    os.makedirs(out_dir, exist_ok=True)

    plots_dir = f"{out_dir}/plots"
    if plot_regression:
        os.makedirs(plots_dir, exist_ok=True)

    data_df = pd.read_pickle(data_df_path)

    # Proportion of coordination above 0.5
    data_df["order"] = data_df['trial'].apply(lambda trial_number: 1 if int(trial_number[1:]) % 2 != 0 else 2)
    data_df["mean_second_half"] = data_df['means'].apply(lambda m: np.mean(m[570:]))
    data_df["variance_second_half"] = data_df['means'].apply(lambda m: np.var(m[570:]))
    data_df["prop"] = data_df['means'].apply(lambda m: np.sum(m > 0.5) / len(m))

    # Fixed coordination on second half
    data_df.loc[(data_df["fixed_coordination_second_half"]), "mean_second_half"] = None
    data_df.loc[(data_df["fixed_coordination_second_half"]), "variance_second_half"] = None
    data_df.loc[(data_df["fixed_coordination_second_half"]), "prop"] = None

    result_table = {
        "estimation_name": [],
        "vocalics_aggregation": [],
        "measure": [],
        "order": [],
        "slope": [],
        "intercept": [],
        "pearson": []
    }

    for estimation_name in data_df["estimation_name"].unique():
        if "discrete" in estimation_name:
            measures = ["mean", "mean_second_half", "prop"]
            xlims = [[0, 1], [0, 1], [0, 1]]
        else:
            measures = ["mean", "variance", "mean_second_half", "variance_second_half", "prop"]
            xlims = [[0, 1], [0, 0.2], [0, 1], [0, 0.2], [0, 1]]
        for i, measure in enumerate(measures):
            for aggregation in ["mean", "variance"]:
                data_estimation_df = data_df[
                    (data_df["estimation_name"] == estimation_name) & (data_df["vocalics_aggregation"] == aggregation)]
                data_estimation_df = data_estimation_df[~data_estimation_df["trial"].isin(
                    ["T000719", "T000720", "T000766", "T000798", "T000821", "T000822", "T000853", "T000854"])]
                if not data_estimation_df[measure].isnull().any():
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                    for order in [1, 2]:
                        data_order_df = data_estimation_df[data_estimation_df["order"] == order]
                        xs = data_order_df[measure]
                        ys = data_order_df["score"]

                        result = linregress(xs, ys)
                        result_table["estimation_name"].append(estimation_name)
                        result_table["vocalics_aggregation"].append(aggregation)
                        result_table["measure"].append(measure)
                        result_table["order"].append(order)
                        result_table["slope"].append(result.slope)
                        result_table["intercept"].append(result.intercept)
                        result_table["pearson"].append(result.rvalue)

                        if plot_regression:
                            xs_line = np.array([0, 1])
                            ys_line = result.slope * xs_line + result.intercept

                            # Points
                            axs[order - 1].scatter(xs, ys, color="tab:blue")

                            # Regression Line
                            label = f"$s = {result.slope:.2f}c + {result.intercept:.2f}$. $r = {result.rvalue:.2f}$"
                            axs[order - 1].plot(xs_line, ys_line, color="tab:red", label=label)

                            # Trial Numbers
                            for trial in data_order_df["trial"]:
                                x_text = data_order_df[data_order_df["trial"] == trial][measure]
                                y_text = data_order_df[data_order_df["trial"] == trial]["score"]
                                axs[order - 1].annotate(trial, (x_text, y_text))

                            axs[order - 1].set_xlabel(f"Coordination - {measure}")
                            axs[order - 1].set_ylabel("Final Team Score")
                            axs[order - 1].set_xlim(xlims[i])
                            axs[order - 1].set_ylim([0, 950])
                            axs[order - 1].set_title(f"Mission {order}")
                            axs[order - 1].legend()

                    if plot_regression:
                        plot_filepath = f"{plots_dir}/{aggregation}/{estimation_name}/{measure}.png"
                        os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
                        plt.savefig(plot_filepath, format="png")
                        plt.close(fig)

    df = pd.DataFrame(result_table)
    df.to_csv(f"{out_dir}/result_table.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimates coordination and plot estimates from a list of trials."
    )
    parser.add_argument("--data_df_path", type=str, required=True,
                        help="Filepath to a pandas data frame in .pkl format containing the data.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where estimates must be saved.")
    parser.add_argument("--plot_regression", action="store_true", required=False, default=False,
                        help="Whether plots must be generated for linear regression.")
    args = parser.parse_args()

    estimate(args.data_df_path, args.out_dir, args.plot_regression)
