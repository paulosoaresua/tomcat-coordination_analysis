import argparse
from datetime import datetime
import pickle

from coordination.callback.early_stopping_callback import EarlyStoppingCallback
from coordination.common.log import TensorBoardLogger
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationBlendingLatentVocalics


def train(dataset_path: str, num_train_iter: int, patience: int, seed: int, num_jobs: int, initial_coordination: float,
          a_va: float, b_va: float, a_vaa: float, b_vaa: float, a_vo: float, b_vo: float, a_vuc: float, b_vuc: float,
          initial_var_unbounded_coordination: float, initial_var_coordination: float,
          var_unbounded_coordination_proposal: float, var_coordination_proposal: float,
          unbounded_coordination_num_mcmc_iterations: int, coordination_num_mcmc_iterations: int, out_dir: str):
    # Loading dataset
    with open(dataset_path, "rb") as f:
        dataset = BetaCoordinationLatentVocalicsDataset.from_latent_vocalics_dataset(pickle.load(f))

    timestamp = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    out_dir = f"{out_dir}/{timestamp}"

    model = BetaCoordinationBlendingLatentVocalics(
        initial_coordination=initial_coordination,
        num_vocalic_features=dataset.series[0].num_vocalic_features,
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

    tb_logger = TensorBoardLogger(f"{out_dir}/tensorboard")

    if patience > 0:
        callbacks = [EarlyStoppingCallback(patience=patience)]
    else:
        callbacks = []

    model.reset_parameters()
    model.fit(evidence=dataset,
              burn_in=num_train_iter,
              num_jobs=num_jobs,
              seed=seed,
              logger=tb_logger,
              callbacks=callbacks)

    with open(f"{out_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a coordination model on a dataset of observed vocalic features over time."
    )

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset used to train the coordination model.")
    parser.add_argument("--n_train_iter", type=int, required=True, help="Number of training iterations.")
    parser.add_argument("--patience", type=int, required=True, default=5,
                        help="Number of iterations with no improvement to wait before stopping training. If <= 0, it trains for the full amount of iterations.")
    parser.add_argument("--seed", type=int, required=False, default=0, help="Random seed for replication.")
    parser.add_argument("--n_jobs", type=int, required=False, default=1, help="Number of jobs to train in parallel.")
    parser.add_argument("--c0", type=int, required=False, default=0.01, help="Coordination at time 0.")
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
    parser.add_argument("--n_uc_mcmc_iter", type=int, required=False, default=100,
                        help="Number of burn-in iterations when sampling unbounded coordination from its posterior.")
    parser.add_argument("--n_c_mcmc_iter", type=int, required=False, default=100,
                        help="Number of burn-in iterations when sampling coordination from its posterior.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where to save the model.")

    args = parser.parse_args()

    train(dataset_path=args.dataset_path,
          num_train_iter=args.n_train_iter,
          patience=args.patience,
          seed=args.seed,
          num_jobs=args.n_jobs,
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
