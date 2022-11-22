import numpy as np
from sklearn.model_selection import KFold

import argparse
from copy import deepcopy
from datetime import datetime
import pickle

from tqdm import tqdm

from coordination.common.log import TensorBoardLogger
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationBlendingLatentVocalics


def create_splits(num_train_val_splits: int, num_train_test_splits: int, num_train_iter: int, num_jobs_split: int,
                  num_jobs_train: int, initial_coordination: float, a_va: float, b_va: float, a_vaa: float,
                  b_vaa: float, a_vo: float, b_vo: float, a_vuc: float, b_vuc: float,
                  initial_var_unbounded_coordination: float, initial_var_coordination: float,
                  var_unbounded_coordination_proposal: float, var_coordination_proposal: float,
                  unbounded_coordination_num_mcmc_iterations: int, coordination_num_mcmc_iterations: int, out_dir: str):
    """
    We execute a nested procedure for cross validation. First, we split the training and test data into folds.
    For each training fold, we split that into more folds, creating several training and validation groups. This
    is necessary because to train the regression model using the inferred coordination values we need an independent
    validation set.
    """

    timestamp = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    out_dir = f"{out_dir}/{timestamp}"

    model = BetaCoordinationBlendingLatentVocalics(
        initial_coordination=initial_coordination,
        num_vocalic_features=2,
        num_speakers=3,
        a_va=a_va,
        b_va=b_va,
        a_vaa=a_vaa,
        b_vaa=b_vaa,
        a_vo=a_vo,
        b_vo=b_vo,
        a_vuc=a_vuc,
        b_vuc=b_vuc,
        initial_var_unbounded_coordination=initial_var_unbounded_coordination,
        initial_var_coordination=initial_var_coordination,
        var_unbounded_coordination_proposal=var_unbounded_coordination_proposal,
        var_coordination_proposal=var_coordination_proposal,
        unbounded_coordination_num_mcmc_iterations=unbounded_coordination_num_mcmc_iterations,
        coordination_num_mcmc_iterations=coordination_num_mcmc_iterations
    )

    model.var_uc = 0.25
    model.var_cc = 1e-6
    model.var_a = 1
    model.var_aa = 0.25
    model.var_o = 1
    samples = model.sample(100, 50, seed=0, time_scale_density=1)
    model.reset_parameters()

    evidence = BetaCoordinationLatentVocalicsDataset.from_samples(samples)
    evidence.remove_all(["unbounded_coordination", "coordination", "latent_vocalics"])

    train_val_splitter = KFold(n_splits=num_train_val_splits, shuffle=True)
    train_test_splitter = KFold(n_splits=num_train_test_splits, shuffle=True)

    results = {
        "outer_splits": []
    }

    X = np.arange(evidence.num_trials)[:, np.newaxis]
    for outer_split_num, outer_indices in tqdm(enumerate(train_test_splitter.split(X)), "Outer Split", position=-2):
        train_indices, test_indices = outer_indices
        # Train on all the training samples in the outer split
        tb_logger = TensorBoardLogger(f"{out_dir}/logs/outer_split_{outer_split_num}/events")
        tb_logger.add_info("outer_split", outer_split_num)
        tb_logger.add_info("train_indices", train_indices.tolist())
        tb_logger.add_info("test_indices", test_indices.tolist())

        model.reset_parameters()
        model.fit(evidence=evidence.get_subset(train_indices),
                  burn_in=num_train_iter,
                  num_jobs=num_jobs_train,
                  seed=0,
                  logger=tb_logger)

        outer_split_result = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "model": deepcopy(model),
            "inner_splits": []
        }

        for inner_split_num, inner_indices in tqdm(
                enumerate(train_val_splitter.split(train_indices[:, np.newaxis])), position=-1, leave=False):
            inner_train_indices, val_indices = inner_indices
            tb_logger = TensorBoardLogger(
                f"{out_dir}/logs/outer_split_{outer_split_num}/inner_split_{inner_split_num}/events")
            tb_logger.add_info("outer_split", outer_split_num)
            tb_logger.add_info("inner_split", inner_split_num)
            tb_logger.add_info("train_indices", inner_train_indices.tolist())
            tb_logger.add_info("val_indices", val_indices.tolist())

            model.reset_parameters()
            model.fit(evidence=evidence.get_subset(inner_train_indices),
                      burn_in=num_train_iter,
                      num_jobs=num_jobs_train,
                      seed=0,
                      logger=tb_logger)

            inner_split_result = {
                "train_indices": inner_train_indices,
                "val_indices": val_indices,
                "model": deepcopy(model)
            }

            outer_split_result["inner_splits"].append(inner_split_result)

        results["outer_splits"].append(outer_split_result)

    with open(f"{out_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset of vocalics for mission 1, mission 2 and both missions for all the serialized "
                    "trials in a given folder."
    )

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save the results and logs.")
    parser.add_argument("--n_val_splits", type=int, required=True,
                        help="Number of inner splits for validation.")
    parser.add_argument("--n_test_splits", type=int, required=True,
                        help="Number of outer splits fot test.")
    parser.add_argument("--n_train_iter", type=int, required=True,
                        help="Number of training iterations.")
    parser.add_argument("--n_jobs_split", type=int, required=False, default=1,
                        help="Number of jobs to execute outer splits in parallel.")
    parser.add_argument("--n_jobs_train", type=int, required=False, default=1,
                        help="Number of jobs to train in parallel.")
    parser.add_argument("--c0", type=int, required=False, default=0.01,
                        help="Coordination at time 0.")
    parser.add_argument("--a_vuc", type=float, required=False, default=1,
                        help="1st prior parameter of the variance of the unbounded coordination.")
    parser.add_argument("--b_vuc", type=float, required=False, default=1,
                        help="2nd prior parameter of the variance of the unbounded coordination.")
    parser.add_argument("--a_va", type=float, required=False, default=1,
                        help="1st prior parameter of the variance of initial latent vocalics.")
    parser.add_argument("--b_va", type=float, required=False, default=1,
                        help="2nd prior parameter of the variance of initial latent vocalics.")
    parser.add_argument("--a_vaa", type=float, required=False, default=1,
                        help="1st prior parameter of the variance of latent vocalics.")
    parser.add_argument("--b_vaa", type=float, required=False, default=1,
                        help="2nd prior parameter of the variance of latent vocalics.")
    parser.add_argument("--a_vo", type=float, required=False, default=1,
                        help="1st prior parameter of the variance of the observed vocalics.")
    parser.add_argument("--b_vo", type=float, required=False, default=1,
                        help="2nd prior parameter of the variance of the observed vocalics.")
    parser.add_argument("--vuc0", type=float, required=False, default=1e-4,
                        help="Initial variance of unbounded coordination during training.")
    parser.add_argument("--vcc0", type=float, required=False, default=1e-6,
                        help="Initial variance of coordination during training.")
    parser.add_argument("--vuc_prop", type=float, required=False, default=1e-3,
                        help="Variance of the proposal distribution for unbounded coordination.")
    parser.add_argument("--vcc_prop", type=float, required=False, default=1e-6,
                        help="Variance of the proposal distribution for coordination.")
    parser.add_argument("--n_uc_mcmc_iter", type=int, required=False, default=50,
                        help="Number of burn-in iterations when sampling unbounded coordination from its posterior.")
    parser.add_argument("--n_c_mcmc_iter", type=int, required=False, default=50,
                        help="Number of burn-in iterations when sampling coordination from its posterior.")

    args = parser.parse_args()

    create_splits(num_train_val_splits=args.n_val_splits,
                  num_train_test_splits=args.n_test_splits,
                  num_train_iter=args.n_train_iter,
                  num_jobs_split=args.n_jobs_split,
                  num_jobs_train=args.n_jobs_train,
                  initial_coordination=args.c0,
                  a_va=args.a_va,
                  b_va=args.b_va,
                  a_vaa=args.a_vaa,
                  b_vaa=args.b_vaa,
                  a_vo=args.a_vo,
                  b_vo=args.b_vo,
                  a_vuc=args.a_vuc,
                  b_vuc=args.b_vuc,
                  initial_var_unbounded_coordination=args.vuc0,
                  initial_var_coordination=args.vcc0,
                  var_unbounded_coordination_proposal=args.vuc_prop,
                  var_coordination_proposal=args.vcc_prop,
                  unbounded_coordination_num_mcmc_iterations=args.n_uc_mcmc_iter,
                  coordination_num_mcmc_iterations=args.n_c_mcmc_iter,
                  out_dir=args.out_dir)
