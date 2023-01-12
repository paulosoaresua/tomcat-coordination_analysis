import argparse
from datetime import datetime
from multiprocessing import Pool
import pickle

import numpy as np

from coordination.model.brain_body_model import BrainBodyModel
from coordination.model.utils.brain_body_model import BrainBodyDataset, BrainBodySamples, BrainBodyParticlesSummary


def fit(data_dir: str, initial_coordination: float, burn_in: int, num_samples: int,
        num_chains: int, num_inference_jobs: int, num_trial_jobs: int, seed: int, out_dir: str, ref_date: str):
    """
    We fit parameters and coordination for each individual trial as the parameters of the model might vary per team.
    """

    # Loading dataset
    samples = BrainBodySamples()
    with open(f"{data_dir}/brain_signals.pkl", "rb") as f:
        samples.observed_brain = pickle.load(f)

    with open(f"{data_dir}/body_movements.pkl", "rb") as f:
        samples.observed_body = pickle.load(f)

    evidence = BrainBodyDataset.from_samples(samples)
    evidence.normalize_per_subject()

    if ref_date is None or len(ref_date) == 0:
        ref_date = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")

    out_dir = f"{out_dir}/{ref_date}"

    # Fit and save each one of the trials
    inference_data = []
    inference_summaries = []
    with Pool(num_trial_jobs) as pool:
        trial_blocks = np.array_split(np.arange(evidence.num_trials), num_trial_jobs)
        job_args = [(initial_coordination, evidence.get_subset(block), burn_in, num_samples, num_chains,
                     num_inference_jobs, seed) for block in trial_blocks]

        results = pool.starmap(fit_helper, job_args)
        for idata, isummaries in results:
            inference_data.extend(idata)
            inference_summaries.extend(isummaries)

    with open(f"{out_dir}/inference_data.pkl", "wb") as f:
        pickle.dump(inference_data, f)

    with open(f"{out_dir}/inference_summaries.pkl", "wb") as f:
        pickle.dump(inference_summaries, f)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a coordination model on a dataset of observed vocalic features over time."
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing brain_signals.pkl and body_movements.pkl files as evidence to the "
                             "model.")
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

    args = parser.parse_args()

    fit(data_dir=args.data_dir,
        initial_coordination=args.c0,
        burn_in=args.burn_in,
        num_samples=args.n_samples,
        num_chains=args.n_chains,
        num_jobs=args.n_jobs,
        seed=args.seed,
        out_dir=args.out_dir,
        ref_date=args.ref_date)
