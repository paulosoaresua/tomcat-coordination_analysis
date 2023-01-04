import json
from typing import List

import argparse
from datetime import datetime
import pickle

import numpy as np
from sklearn.model_selection import KFold

from coordination.callback.early_stopping_callback import EarlyStoppingCallback
from coordination.common.log import TensorBoardLogger
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
from coordination.model.gendered_beta_coordination_blending_latent_vocalics import \
    GenderedBetaCoordinationBlendingLatentVocalics
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsTrainingHyperParameters
from coordination.model.utils.gendered_beta_coordination_blending_latent_vocalics import \
    GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters


def train(dataset_path: str, num_train_iter: int, patience: int, seed: int, num_jobs: int, c0: float, a_vu: float,
          b_vu: float, a_va: float, b_va: float, a_vaa: float, b_vaa: float, a_vo: float, b_vo: float,
          mu_mo_male: float, nu_mo_male: float, a_vo_male: float, b_vo_male: float, mu_mo_female: float,
          nu_mo_female: float, a_vo_female: float, b_vo_female: float, vu0: float, vc0: float, va0: float, vaa0: float,
          mo0_male: np.ndarray, mo0_female: np.ndarray, vo0_male: np.ndarray, vo0_female: np.ndarray, vo0: float,
          u_mcmc_iter: int, c_mcmc_iter: int, vu_mcmc_prop: float, vc_mcmc_prop: float, out_dir: str,
          no_self_dependency: bool, features: List[str], gendered: bool, cv: int, no_link: bool, ref_date: str):

    num_features = len(features)

    assert len(mo0_male) == 1 or len(mo0_male) == num_features
    assert len(mo0_female) == 1 or len(mo0_male) == num_features
    assert len(vo0_male) == 1 or len(mo0_male) == num_features
    assert len(vo0_female) == 1 or len(mo0_male) == num_features
    assert cv >= 1

    if len(mo0_male) == 1:
        mo0_male = np.ones(num_features) * mo0_male[0]

    if len(mo0_female) == 1:
        mo0_female = np.ones(num_features) * mo0_female[0]

    if len(vo0_male) == 1:
        vo0_male = np.ones(num_features) * vo0_male[0]

    if len(vo0_female) == 1:
        vo0_female = np.ones(num_features) * vo0_female[0]

    # Loading dataset
    with open(dataset_path, "rb") as f:
        dataset = BetaCoordinationLatentVocalicsDataset.from_latent_vocalics_dataset(pickle.load(f))

    dataset.keep_vocalic_features(features)

    if no_link:
        dataset.disable_speech_semantic_links()

    if gendered:
        dataset.normalize_gender()

        train_hyper_parameters = GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters(
            a_vu=a_vu,
            b_vu=b_vu,
            a_va=a_va,
            b_va=b_va,
            a_vaa=a_vaa,
            b_vaa=b_vaa,
            mu_mo_male=mu_mo_male,
            nu_mo_male=nu_mo_male,
            a_vo_male=a_vo_male,
            b_vo_male=b_vo_male,
            mu_mo_female=mu_mo_female,
            nu_mo_female=nu_mo_female,
            a_vo_female=a_vo_female,
            b_vo_female=b_vo_female,
            vu0=vu0,
            vc0=vc0,
            va0=va0,
            vaa0=vaa0,
            mo0_male=mo0_male,
            mo0_female=mo0_female,
            vo0_male=vo0_male,
            vo0_female=vo0_female,
            u_mcmc_iter=u_mcmc_iter,
            c_mcmc_iter=c_mcmc_iter,
            vu_mcmc_prop=vu_mcmc_prop,
            vc_mcmc_prop=vc_mcmc_prop
        )

        model = GenderedBetaCoordinationBlendingLatentVocalics(
            initial_coordination=c0,
            num_vocalic_features=dataset.series[0].num_vocalic_features,
            num_speakers=3,
            disable_self_dependency=no_self_dependency)
    else:
        dataset.normalize_per_subject()

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
            num_speakers=3,
            disable_self_dependency=no_self_dependency)
    if ref_date is None or len(ref_date) == 0:
        ref_date = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    out_dir = f"{out_dir}/{ref_date}"

    if patience > 0:
        callbacks = [EarlyStoppingCallback(patience=patience)]
    else:
        callbacks = []

    if cv == 1:
        tb_logger = TensorBoardLogger(f"{out_dir}/tensorboard")
        model.reset_parameters()
        model.fit(evidence=dataset,
                  train_hyper_parameters=train_hyper_parameters,
                  burn_in=num_train_iter,
                  num_jobs=num_jobs,
                  seed=seed,
                  logger=tb_logger,
                  callbacks=callbacks)
        model.save(out_dir)
    else:
        train_test_splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
        X = np.arange(dataset.num_trials)[:, np.newaxis]
        for split_num, indices in enumerate(train_test_splitter.split(X)):
            print("")
            print(f"~~~~~~~~~ SPLIT {split_num}/{cv} ~~~~~~~~~")
            print("")

            train_indices, test_indices = indices

            split_out_dir = f"{out_dir}/split_{split_num}"
            tb_logger = TensorBoardLogger(f"{split_out_dir}/tensorboard")
            tb_logger.add_info("split_num", split_num)
            tb_logger.add_info("train_indices", train_indices.tolist())
            tb_logger.add_info("test_indices", test_indices.tolist())

            model.reset_parameters()
            model.fit(evidence=dataset.get_subset(train_indices),
                      train_hyper_parameters=train_hyper_parameters,
                      burn_in=num_train_iter,
                      num_jobs=num_jobs,
                      seed=seed,
                      logger=tb_logger,
                      callbacks=callbacks)
            model.save(split_out_dir)

            split_info = {
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
            }

            with open(f"{split_out_dir}/split_info.json", "w") as f:
                json.dump(split_info, f)


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
    parser.add_argument("--mu_mo_male", type=float, required=False, default=0,
                        help="1st prior parameter of the means of the observed vocalics from males.")
    parser.add_argument("--nu_mo_male", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the means of the observed vocalics from males.")
    parser.add_argument("--a_vo_male", type=float, required=False, default=1e-6,
                        help="1st prior parameter of the variances of the observed vocalics from males.")
    parser.add_argument("--b_vo_male", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the variances of the observed vocalics from males.")
    parser.add_argument("--mu_mo_female", type=float, required=False, default=0,
                        help="1st prior parameter of the means of the observed vocalics from females.")
    parser.add_argument("--nu_mo_female", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the means of the observed vocalics from females.")
    parser.add_argument("--a_vo_female", type=float, required=False, default=1e-6,
                        help="1st prior parameter of the variances of the observed vocalics from females.")
    parser.add_argument("--b_vo_female", type=float, required=False, default=1e-6,
                        help="2nd prior parameter of the variances of the observed vocalics from females.")
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
    parser.add_argument("--mo0_male", type=str, required=False, default="0,0,0,0",
                        help="Initial mean of observed vocalics from males.")
    parser.add_argument("--mo0_female", type=str, required=False, default="0,0,0,0",
                        help="Initial mean of observed vocalics from females.")
    parser.add_argument("--vo0_male", type=str, required=False, default="1,1,1,1",
                        help="Initial variance of observed vocalics from males.")
    parser.add_argument("--vo0_female", type=str, required=False, default="1,1,1,1",
                        help="Initial variance of observed vocalics from females.")
    parser.add_argument("--u_mcmc_iter", type=int, required=False, default=100,
                        help="Number of burn-in iterations when sampling unbounded coordination from its posterior.")
    parser.add_argument("--c_mcmc_iter", type=int, required=False, default=100,
                        help="Number of burn-in iterations when sampling coordination from its posterior.")
    parser.add_argument("--vu_mcmc_prop", type=float, required=False, default=0.001,
                        help="Variance of the proposal distribution for unbounded coordination.")
    parser.add_argument("--vc_mcmc_prop", type=float, required=False, default=0.001,
                        help="Variance of the proposal distribution for coordination.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where to save the model.")
    parser.add_argument("--no_self_dep", action="store_true", required=False, default=False,
                        help="Disable latent vocalics dependency on the same speaker.")
    parser.add_argument("--features", type=str, required=False, default="pitch, intensity, jitter, shimmer",
                        help="List of vocalic features to consider. It can be any subset of the default value.")
    parser.add_argument("--gendered", action="store_true", required=False, default=False,
                        help="Whether to use a model that considers speakers' genders.")
    parser.add_argument("--cv", type=int, required=False, default=1,
                        help="Number of splits if the model is to be trained for cross-validation.")
    parser.add_argument("--no_link", action="store_true", required=False, default=False,
                        help="Whether to disable semantic link.")
    parser.add_argument("--ref_date", type=str, required=False, default="",
                        help="Name of the folder inside out_dir where to save the model and logs. If not informed, the "
                             "program will create a folder with the timestamp at the execution time.")

    args = parser.parse_args()


    def format_feature_name(name: str):
        return name.strip().lower()


    def format_vector_param(name: str):
        return float(name.strip().lower())


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
          mu_mo_male=args.mu_mo_male,
          nu_mo_male=args.nu_mo_male,
          a_vo_male=args.a_vo_male,
          b_vo_male=args.b_vo_male,
          mu_mo_female=args.mu_mo_female,
          nu_mo_female=args.nu_mo_female,
          a_vo_female=args.a_vo_female,
          b_vo_female=args.b_vo_female,
          vc0=args.vc0,
          vu0=args.vu0,
          va0=args.va0,
          vaa0=args.vaa0,
          vo0=args.vo0,
          mo0_male=np.array(list(map(format_vector_param, args.mo0_male.split(",")))),
          mo0_female=np.array(list(map(format_vector_param, args.mo0_female.split(",")))),
          vo0_male=np.array(list(map(format_vector_param, args.vo0_male.split(",")))),
          vo0_female=np.array(list(map(format_vector_param, args.vo0_female.split(",")))),
          u_mcmc_iter=args.u_mcmc_iter,
          c_mcmc_iter=args.c_mcmc_iter,
          vu_mcmc_prop=args.vu_mcmc_prop,
          vc_mcmc_prop=args.vc_mcmc_prop,
          out_dir=args.out_dir,
          no_self_dependency=args.no_self_dep,
          features=list(map(format_feature_name, args.features.split(","))),
          gendered=args.gendered,
          cv=args.cv,
          no_link=args.no_link,
          ref_date=args.ref_date)
