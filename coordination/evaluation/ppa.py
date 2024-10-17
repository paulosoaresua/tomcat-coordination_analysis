import argparse
import os.path
from typing import Optional
import pandas as pd
from coordination.inference.inference_run import InferenceRun
from coordination.common.config import settings
import numpy as np


def evaluate_ppa(inference_run: InferenceRun) -> Optional[pd.DataFrame]:
    """
    Computes the average and standard error PPAs for the experiments in an inference run.

    @param inference_run: inference run to evaluate.
    @return: a DataFrame containing the summarized values.
    """
    dfs = []
    for experiment_id in inference_run.experiment_ids:
        df = inference_run.get_ppa_results(experiment_id)
        if df is None:
            return None

        dfs.append(df)

    def summarize(x):
        """
        String containing mean and standard error of the values.

        @param x: data frame containing the values grouped by non-window columns.
        @return: single dataframe with aggregated metrics.
        """
        window_cols = [c for c in x.columns if c[0] == "w"]
        return pd.Series([f"{x[c].mean():4f} ({(x[c].std() / np.sqrt(len(x[c]))):4f})" for
                          c in window_cols], window_cols)

    df = pd.concat(dfs)
    non_window_cols = [c for c in df.columns if c[0] != "w"]
    return df.groupby(non_window_cols).apply(summarize).reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)

    args = parser.parse_args()

    inference_run = InferenceRun(
        inference_dir=settings.inferences_dir,
        run_id=args.run_id
    )

    df = evaluate_ppa(inference_run)
    if df is not None:
        os.makedirs(os.path.join(settings.evaluations_dir, args.run_id), exist_ok=True)
        df.to_csv(str(os.path.join(settings.evaluations_dir, args.run_id, "ppa.csv")))
