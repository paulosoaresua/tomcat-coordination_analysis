from typing import List

from datetime import datetime
import pickle
import os
import uuid

import numpy as np
import pandas as pd

from coordination.model.brain_model import BrainModel, BrainSeries
from coordination.model.body_model import BodyModel, BodySeries
from coordination.model.brain_body_model import BrainBodyModel, BrainBodySeries
from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries
from coordination.model.vocalic_model import VocalicModel, VocalicSeries


def inference(out_dir: str, execution_time: str, experiment_ids: List[str], evidence_filepath: str, model_name: str,
              burn_in: int, num_samples: int, num_chains: int, seed: int, num_inference_jobs: int,
              initial_coordination: float, num_subjects: int, brain_channels: List[str], vocalic_features: List[str],
              self_dependent: bool, sd_uc: float, sd_mean_a0_brain: np.ndarray, sd_sd_aa_brain: np.ndarray,
              sd_sd_o_brain: np.ndarray, sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray,
              sd_sd_o_body: np.ndarray, a_mixture_weights: np.ndarray, sd_mean_a0_vocalic: np.ndarray,
              sd_sd_aa_vocalic: np.ndarray, sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float,
              b_p_semantic_link: float):
    evidence_df = pd.read_csv(evidence_filepath, index_col=0)
    evidence_df = evidence_df[evidence_df["experiment_id"].isin(experiment_ids)]

    # Create correct model from the provided name
    if model_name == "brain":
        model = BrainModel(initial_coordination=initial_coordination,
                           num_subjects=num_subjects,
                           num_brain_channels=len(brain_channels),
                           self_dependent=self_dependent,
                           sd_uc=sd_uc,
                           sd_mean_a0=sd_mean_a0_brain,
                           sd_sd_aa=sd_sd_aa_brain,
                           sd_sd_o=sd_sd_o_brain,
                           a_mixture_weights=a_mixture_weights)

        evidence = BrainSeries.from_data_frame(evidence_df, brain_channels)
    elif model_name == "body":
        model = BodyModel(initial_coordination=initial_coordination,
                          num_subjects=num_subjects,
                          self_dependent=self_dependent,
                          sd_uc=sd_uc,
                          sd_mean_a0=sd_mean_a0_body,
                          sd_sd_aa=sd_sd_aa_body,
                          sd_sd_o=sd_sd_o_body,
                          a_mixture_weights=a_mixture_weights)

        evidence = BodySeries.from_data_frame(evidence_df)
    elif model_name == "brain_body":
        model = BrainBodyModel(initial_coordination=initial_coordination,
                               num_subjects=num_subjects,
                               num_brain_channels=len(brain_channels),
                               self_dependent=self_dependent,
                               sd_uc=sd_uc,
                               sd_mean_a0_brain=sd_mean_a0_brain,
                               sd_sd_aa_brain=sd_sd_aa_brain,
                               sd_sd_o_brain=sd_sd_o_brain,
                               sd_mean_a0_body=sd_mean_a0_body,
                               sd_sd_aa_body=sd_sd_aa_body,
                               sd_sd_o_body=sd_sd_o_body,
                               a_mixture_weights=a_mixture_weights)

        evidence = BrainBodySeries.from_data_frame(evidence_df, brain_channels)
    elif model_name == "vocalic_semantic":
        model = VocalicSemanticModel(initial_coordination=initial_coordination,
                                     num_subjects=num_subjects,
                                     num_vocalic_features=len(vocalic_features),
                                     self_dependent=self_dependent,
                                     sd_uc=sd_uc,
                                     sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                                     sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                                     sd_sd_o_vocalic=sd_sd_o_vocalic,
                                     a_p_semantic_link=a_p_semantic_link,
                                     b_p_semantic_link=b_p_semantic_link)

        evidence = VocalicSemanticSeries.from_data_frame(evidence_df, vocalic_features)
    elif model_name == "vocalic":
        model = VocalicModel(initial_coordination=initial_coordination,
                             num_subjects=num_subjects,
                             num_vocalic_features=len(vocalic_features),
                             self_dependent=self_dependent,
                             sd_uc=sd_uc,
                             sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                             sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                             sd_sd_o_vocalic=sd_sd_o_vocalic)

        evidence = VocalicSeries.from_data_frame(evidence_df, vocalic_features)
    else:
        raise Exception(f"Invalid model {model}.")

    for single_evidence_series in evidence:
        _, idata = model.prior_predictive(evidence, seed)
        pymc_model, idata_posterior = model.fit(evidence=single_evidence_series,
                                                burn_in=burn_in,
                                                num_samples=num_samples,
                                                num_chains=num_chains,
                                                seed=seed,
                                                num_jobs=num_inference_jobs)
        idata.extend(idata_posterior)

        # current_timestamp = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
        results_folder = f"{out_dir}/{model}/{execution_time}/{single_evidence_series.uuid}"

        # Save execution hyper-parameters
        with open(f"{results_folder}/inference_data.pkl", "wb") as f:
            pickle.dump(idata, f)

        # Save pymc_model used for posterior inference
        with open(f"{results_folder}/pymc_model.pkl", "wb") as f:
            pickle.dump(pymc_model, f)

        # Save inference data
        with open(f"{results_folder}/inference_data.pkl", "wb") as f:
            pickle.dump(idata, f)

        save_predictive_prior_plots(idata, single_evidence_series, model)
        save_coordination_plots(idata)


# Save
# 1. pymc_model
# 2. inference data
# 3. Predictive prior plots
# 4. Coordination posterior plots
# 5. Hyper-parameters


if __name__ == "__main__":
    pass
