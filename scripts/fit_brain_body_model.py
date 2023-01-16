import argparse
import os
from datetime import datetime
from multiprocessing import Pool
import os
import pickle

from ast import literal_eval
import numpy as np
import pandas as pd

from coordination.model.brain_body_model import BrainBodyModel
from coordination.model.utils.brain_body_model import BrainBodyDataset, BrainBodySamples, BrainBodyParticlesSummary


def fit(brain_data_path: str, body_data_path: str, num_time_steps: int, initial_coordination: float, burn_in: int,
        num_samples: int, num_chains: int, num_inference_jobs: int, num_trial_jobs: int, seed: int, out_dir: str,
        ref_date: str, data_idx: int):
    """
    We fit parameters and coordination for each individual trial as the parameters of the model might vary per team.
    """

    # TODO: I am disabling body data for now

    # Loading dataset
    brain_df = pd.read_csv(brain_data_path, index_col=0)
    # body_df = pd.read_csv(body_data_path, index_col=0)

    # TODO: remove the 2 lines below if body is to be included
    body_df = brain_df.copy()
    body_df.columns = [c if c != "avg_hb_total" else "total_energy" for c in brain_df.columns]
    data_df = brain_df.merge(body_df, on=["experiment_id", "subject"], suffixes=("_brain", "_body"))

    # TODO: implement the use of masks in the future to deal with brain, body and coordination at different scales
    brain_signals, _, body_movements, _, experiments = data_frame_to_ndarrays(data_df, num_time_steps)

    samples = BrainBodySamples()
    if data_idx > -1:
        # Specific trial in execution
        samples.observed_brain = brain_signals[data_idx][None, :]
        samples.observed_body = body_movements[data_idx][None, :]
    else:
        samples.observed_brain = brain_signals
        samples.observed_body = body_movements

    evidence = BrainBodyDataset.from_samples(samples)
    # evidence.normalize_per_subject()

    if ref_date is None or len(ref_date) == 0:
        ref_date = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")

    out_dir = f"{out_dir}/{ref_date}"

    # Fit and save each one of the trials
    inference_data = []
    result_table = []
    for i in range(evidence.num_trials):
        print("")
        print(f"Trial {i + 1}/{evidence.num_trials}: {experiments[i]}")

        idata, isummaries = fit_helper(evidence=evidence.get_subset([i]),
                                       initial_coordination=initial_coordination,
                                       burn_in=burn_in,
                                       num_samples=num_samples,
                                       num_chains=num_chains,
                                       num_jobs=num_inference_jobs,
                                       seed=seed)

        inference_data.extend(idata)
        result_table.append(
            [experiments[i], isummaries[0].coordination_mean.tolist(), isummaries[0].coordination_std.tolist(),
             isummaries[0].coordination_mean.mean(), isummaries[0].coordination_std.mean()])

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/inference_data.pkl", "wb") as f:
        pickle.dump(inference_data, f)

    df = pd.DataFrame(result_table,
                      columns=["experiment_id", "coordination_means", "coordination_stds", "avg_coordination_mean",
                               "avg_coordination_std"])
    df.to_csv(f"{out_dir}/inference_table.csv")


def fit_helper(evidence: BrainBodyDataset, initial_coordination: float, burn_in: int, num_samples: int, num_chains: int,
               num_jobs: int, seed: int):
    model = BrainBodyModel(
        initial_coordination=initial_coordination,
        num_brain_channels=evidence.observed_brain_signals.shape[2],
        num_subjects=evidence.observed_brain_signals.shape[1]
    )

    inference_data = []
    inference_summaries = []
    for trial in range(evidence.num_trials):
        model.parameters.reset()
        idata = model.fit(evidence=evidence.get_subset([trial]),
                          burn_in=burn_in,
                          num_samples=num_samples,
                          num_chains=num_chains,
                          num_jobs=num_jobs,
                          seed=seed)

        inference_data.append(idata)
        inference_summaries.append(BrainBodyParticlesSummary.from_inference_data(idata))

    return inference_data, inference_summaries


def vec_linspace(start, stop, N):
    """
    Elementwise linear interpolation between 2 vectors.
    """
    steps = (1.0 / N) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]


def fill_gaps(data, missing_mask):
    """
    Fill time steps with no observation with a linear interpolation of the two closest observed points in time.
    """

    num_time_steps = data.shape[1]

    next_index_with_value = np.zeros(num_time_steps)
    for t in range(num_time_steps - 1, -1, -1):
        next_index_with_value[t] = t if missing_mask[t] == 1 or t == num_time_steps - 1 else next_index_with_value[
            t + 1]

    t = 0
    while t < num_time_steps:
        if missing_mask[t] == 0:
            next_t = int(next_index_with_value[t])

            if t == 0:
                # Repeat the values
                if next_t == 1:
                    data[:, 0:next_t] = data[:, next_t]
                else:
                    data[:, 0:next_t] = data[:, next_t][:, None]
            elif t == num_time_steps - 1:
                # Repeat the values
                data[:, t] = data[:, t - 1]
            else:
                gap_size = next_t - t
                data[:, t:next_t] = vec_linspace(data[:, t - 1], data[:, next_t], gap_size + 1)[:, 1:]

            t = next_t + 1
        else:
            t += 1

    return data


def to_coordination_timescale(data, num_time_steps, data_frequency):
    """
    Fit observation in a sparse array that matches coordination timescale.
    """
    coordination_frequency = (num_time_steps / data.shape[1]) * data_frequency

    assert coordination_frequency >= data_frequency

    values = np.zeros((data.shape[0], num_time_steps))
    mask = np.zeros(num_time_steps)
    for t in range(num_time_steps):
        if (t * data_frequency) % coordination_frequency == 0:
            # Match between coordination time scale and data time scale
            t_data = int(t * data_frequency / coordination_frequency)

            values[:, t] = data[:, t_data]
            mask[t] = 1

    return values, mask


def data_frame_to_ndarrays(data_df: pd.DataFrame, num_time_steps: int):
    all_brain_signals = []
    all_brain_masks = []
    all_body_movements = []
    all_body_masks = []
    experiments = sorted(list(data_df["experiment_id"].unique()))

    for experiment_id in experiments:
        brain_signals_per_subject = []
        body_movements_per_subject = []

        for subject in ["lion", "tiger", "leopard"]:
            row = data_df[(data_df["experiment_id"] == experiment_id) & (data_df["subject"] == subject)]

            # Brain signal
            brain_signals = np.array(literal_eval(row["avg_hb_total"].values[0]))
            num_measurements = np.array(literal_eval(row["measurements_per_window_brain"].values[0]))
            missing_data = np.where(num_measurements == 0, 0, 1)
            brain_signals = fill_gaps(brain_signals, missing_data)
            brain_signals, brain_masks = to_coordination_timescale(brain_signals, num_time_steps,
                                                                   row["frequency_hz_brain"].values[0])
            brain_signals_per_subject.append(brain_signals)

            # Body movements
            body_movements = np.array(literal_eval(row["total_energy"].values[0]))
            num_measurements = np.array(literal_eval(row["measurements_per_window_body"].values[0]))
            missing_data = np.where(num_measurements == 0, 0, 1)
            body_movements = fill_gaps(body_movements, missing_data)
            body_movements, body_masks = to_coordination_timescale(body_movements, num_time_steps,
                                                                   row["frequency_hz_body"].values[0])
            body_movements_per_subject.append(body_movements)

        all_brain_signals.append(brain_signals_per_subject)
        all_brain_masks.append(brain_masks)

        all_body_movements.append(body_movements_per_subject)
        all_body_masks.append(body_masks)

    brain_signals = np.array(all_brain_signals)
    brain_masks = np.array(all_brain_masks)

    body_movements = np.array(all_body_movements)
    body_masks = np.array(all_body_masks)

    return brain_signals, brain_masks, body_movements, body_masks, experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a coordination model on a dataset of observed vocalic features over time."
    )

    parser.add_argument("--brain_data", type=str, required=True,
                        help="Path to the .csv file containing brain data.")
    parser.add_argument("--body_data", type=str, required=True,
                        help="Path to the .csv file containing body data.")
    parser.add_argument("--n_time_steps", type=int, required=True,
                        help="Number of time steps at the coordination scale.")
    parser.add_argument("--c0", type=float, required=True, help="Assumed initial coordination value.")
    parser.add_argument("--burn_in", type=int, required=True, help="Number of discarded samples per chain.")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples per chain.")
    parser.add_argument("--n_chains", type=int, required=True, help="Number of independent chains.")
    parser.add_argument("--n_i_jobs", type=int, required=False, default=1, help="Number of jobs during inference.")
    parser.add_argument("--n_t_jobs", type=int, required=False, default=1, help="Number of jobs to split the trials.")
    parser.add_argument("--seed", type=int, required=False, default=0, help="Random seed for replication.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where to save the model.")
    parser.add_argument("--ref_date", type=str, required=False, default="",
                        help="Name of the folder inside out_dir where to save inference artifacts. If not informed, the "
                             "program will create a folder with the timestamp at the execution time.")
    parser.add_argument("--data_idx", type=int, required=False, default=-1,
                        help="Data index. If different than -1, it will perform inference in the experiment at the informed index. "
                             "It can be used to execute inferences on different trials in parallel since PyMC can not be spawned in parallel "
                             "by the main process.")

    args = parser.parse_args()

    fit(brain_data_path=args.brain_data,
        body_data_path=args.body_data,
        num_time_steps=args.n_time_steps,
        initial_coordination=args.c0,
        burn_in=args.burn_in,
        num_samples=args.n_samples,
        num_chains=args.n_chains,
        num_inference_jobs=args.n_i_jobs,
        num_trial_jobs=args.n_t_jobs,
        seed=args.seed,
        out_dir=args.out_dir,
        ref_date=args.ref_date,
        data_idx=args.data_idx)
