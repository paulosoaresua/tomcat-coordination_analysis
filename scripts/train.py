import argparse
from datetime import datetime
import pickle

from coordination.callback.early_stopping_callback import EarlyStoppingCallback
from coordination.common.log import TensorBoardLogger
import coordination.common.parallelism as parallelism
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsTrainingHyperParameters


def train(dataset_path: str, num_train_iter: int, patience: int, seed: int, num_jobs: int, c0: float, a_vu: float,
          b_vu: float, a_va: float, b_va: float, a_vaa: float, b_vaa: float, a_vo: float, b_vo: float,
          vu0: float, vc0: float, va0: float, vaa0: float, vo0: float, u_mcmc_iter: int, c_mcmc_iter: int,
          vu_mcmc_prop: float, vc_mcmc_prop: float, out_dir: str):

    # Loading dataset
    with open(dataset_path, "rb") as f:
        dataset = BetaCoordinationLatentVocalicsDataset.from_latent_vocalics_dataset(pickle.load(f))

    timestamp = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    out_dir = f"{out_dir}/{timestamp}"

    train_hyper_parameters = BetaCoordinationLatentVocalicsTrainingHyperParameters(
        a_vu=a_vu,
        b_vu=b_vu,
        a_va=a_va,
        b_va=b_va,
        a_vaa=a_vaa,
        b_vaa=b_vaa,
        a_vo=a_vo,
        b_vo=b_vo,
        vu0=vu0,
        vc0=vc0,
        va0=va0,
        vaa0=vaa0,
        vo0=vo0,
        u_mcmc_iter=u_mcmc_iter,
        c_mcmc_iter=c_mcmc_iter,
        vu_mcmc_prop=vu_mcmc_prop,
        vc_mcmc_prop=vc_mcmc_prop
    )

    model = BetaCoordinationBlendingLatentVocalics(
        initial_coordination=c0,
        num_vocalic_features=dataset.series[0].num_vocalic_features,
        num_speakers=3)

    tb_logger = TensorBoardLogger(f"{out_dir}/tensorboard")

    if patience > 0:
        callbacks = [EarlyStoppingCallback(patience=patience)]
    else:
        callbacks = []

    model.reset_parameters()
    model.fit(evidence=dataset,
              train_hyper_parameters=train_hyper_parameters,
              burn_in=num_train_iter,
              num_jobs=num_jobs,
              seed=seed,
              logger=tb_logger,
              callbacks=callbacks)
    model.save(out_dir)


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
    parser.add_argument("--a_vu", type=float, required=False, default=1e-6,
                        help="1st prior parameter of the variance of the unbounded coordination.")
    parser.add_argument("--b_vu", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the variance of the unbounded coordination.")
    parser.add_argument("--a_va", type=float, required=False, default=1e-6,
                        help="1st prior parameter of the variance of initial latent vocalics.")
    parser.add_argument("--b_va", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the variance of initial latent vocalics.")
    parser.add_argument("--a_vaa", type=float, required=False, default=1e-6,
                        help="1st prior parameter of the variance of latent vocalics.")
    parser.add_argument("--b_vaa", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the variance of latent vocalics.")
    parser.add_argument("--a_vo", type=float, required=False, default=1e-6,
                        help="1st prior parameter of the variance of the observed vocalics.")
    parser.add_argument("--b_vo", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the variance of the observed vocalics.")
    parser.add_argument("--vu0", type=float, required=False, default=0.001,
                        help="Initial variance of unbounded coordination during training.")
    parser.add_argument("--vc0", type=float, required=False, default=0.001,
                        help="Initial variance of coordination during training.")
    parser.add_argument("--va0", type=float, required=False, default=1,
                        help="Initial variance of latent vocalics prior.")
    parser.add_argument("--vaa0", type=float, required=False, default=1,
                        help="Initial variance of latent vocalics transition.")
    parser.add_argument("--vo0", type=float, required=False, default=1,
                        help="Initial variance of latent vocalics emission.")
    parser.add_argument("--u_mcmc_iter", type=int, required=False, default=100,
                        help="Number of burn-in iterations when sampling unbounded coordination from its posterior.")
    parser.add_argument("--c_mcmc_iter", type=int, required=False, default=100,
                        help="Number of burn-in iterations when sampling coordination from its posterior.")
    parser.add_argument("--vu_mcmc_prop", type=float, required=False, default=0.001,
                        help="Variance of the proposal distribution for unbounded coordination.")
    parser.add_argument("--vc_mcmc_prop", type=float, required=False, default=0.001,
                        help="Variance of the proposal distribution for coordination.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where to save the model.")

    args = parser.parse_args()

    train(dataset_path=args.dataset_path,
          num_train_iter=args.n_train_iter,
          patience=args.patience,
          seed=args.seed,
          num_jobs=args.n_jobs,
          c0=args.c0,
          a_vu=args.a_vu,
          b_vu=args.b_vu,
          a_va=args.a_va,
          b_va=args.b_va,
          a_vaa=args.a_vaa,
          b_vaa=args.b_vaa,
          a_vo=args.a_vo,
          b_vo=args.b_vo,
          vc0=args.vc0,
          vu0=args.vu0,
          va0=args.va0,
          vaa0=args.vaa0,
          vo0=args.vo0,
          u_mcmc_iter=args.u_mcmc_iter,
          c_mcmc_iter=args.c_mcmc_iter,
          vu_mcmc_prop=args.vu_mcmc_prop,
          vc_mcmc_prop=args.vc_mcmc_prop,
          out_dir=args.out_dir)
