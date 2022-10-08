import json
from typing import List, Optional

import argparse
from glob import glob
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent, VocalicsSparseSeries
from coordination.entity.trial import Trial
from coordination.inference.discrete_coordination import DiscreteCoordinationInferenceFromVocalics
from coordination.inference.logistic_coordination import LogisticCoordinationInferenceFromVocalics
from coordination.inference.truncated_gaussian_coordination_blending import \
    TruncatedGaussianCoordinationBlendingInference
from coordination.inference.truncated_gaussian_coordination_mixture import TruncatedGaussianCoordinationMixtureInference
from coordination.inference.truncated_gaussian_coordination_blending_latent_vocalics import \
    TruncatedGaussianCoordinationBlendingInferenceLatentVocalics
from coordination.plot.coordination import add_discrete_coordination_bar
from scripts.utils import configure_log

ANTI_PHASE_FUNCTION = lambda x, s: -x if s == 0 else x
EITHER_PHASE_FUNCTION = lambda x, s: np.abs(x)


def get_discrete_inference_engines(vocalic_series: VocalicsSparseSeries, p_prior_coordination: float,
                                   p_coordination_transition: float, mean_prior_vocalics: np.ndarray,
                                   std_prior_vocalics: np.ndarray, std_uncoordinated_vocalics: np.ndarray,
                                   std_coordinated_vocalics: np.ndarray):
    inference_engines = [
        (
            "discrete_in_phase_fixed_second_half",
            "last",
            DiscreteCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      p_prior_coordination=p_prior_coordination,
                                                      p_coordination_transition=p_coordination_transition,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      fix_coordination_on_second_half=True)
        ),
        (
            "discrete_anti_phase_fixed_second_half",
            "last",
            DiscreteCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      p_prior_coordination=p_prior_coordination,
                                                      p_coordination_transition=p_coordination_transition,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      f=ANTI_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=True)
        ),
        (
            "discrete_either_phase_fixed_second_half",
            "last",
            DiscreteCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      p_prior_coordination=p_prior_coordination,
                                                      p_coordination_transition=p_coordination_transition,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      f=EITHER_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=True)
        ),
        (
            "discrete_in_phase_variable",
            "all",
            DiscreteCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      p_prior_coordination=p_prior_coordination,
                                                      p_coordination_transition=p_coordination_transition,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      fix_coordination_on_second_half=False)
        ),
        (
            "discrete_anti_phase_variable",
            "all",
            DiscreteCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      p_prior_coordination=p_prior_coordination,
                                                      p_coordination_transition=p_coordination_transition,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      f=ANTI_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=False)
        ),
        (
            "discrete_either_phase_variable",
            "all",
            DiscreteCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      p_prior_coordination=p_prior_coordination,
                                                      p_coordination_transition=p_coordination_transition,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      f=EITHER_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=False)
        )
    ]

    return inference_engines


def get_truncated_gaussian_blending_inference_engines(vocalic_series: VocalicsSparseSeries,
                                                      mean_prior_coordination: float,
                                                      std_prior_coordination: float, std_coordination_drifting: float,
                                                      mean_prior_vocalics: np.ndarray, std_prior_vocalics: np.ndarray,
                                                      std_coordinated_vocalics: np.ndarray):
    inference_engines = [
        (
            "continuous_in_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationBlendingInference(vocalic_series=vocalic_series,
                                                           mean_prior_coordination=mean_prior_coordination,
                                                           std_prior_coordination=std_prior_coordination,
                                                           std_coordination_drifting=std_coordination_drifting,
                                                           mean_prior_vocalics=mean_prior_vocalics,
                                                           std_prior_vocalics=std_prior_vocalics,
                                                           std_coordinated_vocalics=std_coordinated_vocalics,
                                                           fix_coordination_on_second_half=True)
        ),
        (
            "continuous_anti_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationBlendingInference(vocalic_series=vocalic_series,
                                                           mean_prior_coordination=mean_prior_coordination,
                                                           std_prior_coordination=std_prior_coordination,
                                                           std_coordination_drifting=std_coordination_drifting,
                                                           mean_prior_vocalics=mean_prior_vocalics,
                                                           std_prior_vocalics=std_prior_vocalics,
                                                           std_coordinated_vocalics=std_coordinated_vocalics,
                                                           f=ANTI_PHASE_FUNCTION,
                                                           fix_coordination_on_second_half=True)
        ),
        (
            "continuous_either_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationBlendingInference(vocalic_series=vocalic_series,
                                                           mean_prior_coordination=mean_prior_coordination,
                                                           std_prior_coordination=std_prior_coordination,
                                                           std_coordination_drifting=std_coordination_drifting,
                                                           mean_prior_vocalics=mean_prior_vocalics,
                                                           std_prior_vocalics=std_prior_vocalics,
                                                           std_coordinated_vocalics=std_coordinated_vocalics,
                                                           f=EITHER_PHASE_FUNCTION,
                                                           fix_coordination_on_second_half=True)
        ),
        (
            "continuous_in_phase_variable",
            "all",
            TruncatedGaussianCoordinationBlendingInference(vocalic_series=vocalic_series,
                                                           mean_prior_coordination=mean_prior_coordination,
                                                           std_prior_coordination=std_prior_coordination,
                                                           std_coordination_drifting=std_coordination_drifting,
                                                           mean_prior_vocalics=mean_prior_vocalics,
                                                           std_prior_vocalics=std_prior_vocalics,
                                                           std_coordinated_vocalics=std_coordinated_vocalics,
                                                           fix_coordination_on_second_half=False)
        ),
        (
            "continuous_anti_phase_variable",
            "all",
            TruncatedGaussianCoordinationBlendingInference(vocalic_series=vocalic_series,
                                                           mean_prior_coordination=mean_prior_coordination,
                                                           std_prior_coordination=std_prior_coordination,
                                                           std_coordination_drifting=std_coordination_drifting,
                                                           mean_prior_vocalics=mean_prior_vocalics,
                                                           std_prior_vocalics=std_prior_vocalics,
                                                           std_coordinated_vocalics=std_coordinated_vocalics,
                                                           f=ANTI_PHASE_FUNCTION,
                                                           fix_coordination_on_second_half=False)
        ),
        (
            "continuous_either_phase_variable",
            "all",
            TruncatedGaussianCoordinationBlendingInference(vocalic_series=vocalic_series,
                                                           mean_prior_coordination=mean_prior_coordination,
                                                           std_prior_coordination=std_prior_coordination,
                                                           std_coordination_drifting=std_coordination_drifting,
                                                           mean_prior_vocalics=mean_prior_vocalics,
                                                           std_prior_vocalics=std_prior_vocalics,
                                                           std_coordinated_vocalics=std_coordinated_vocalics,
                                                           f=EITHER_PHASE_FUNCTION,
                                                           fix_coordination_on_second_half=False)
        )
    ]

    return inference_engines


def get_logistic_inference_engines(vocalic_series: VocalicsSparseSeries, mean_prior_coordination_logit: float,
                                   std_prior_coordination_logit: float, std_coordination_logit_drifting: float,
                                   mean_prior_vocalics: np.ndarray, std_prior_vocalics: np.ndarray,
                                   std_coordinated_vocalics: np.ndarray, num_particles: int):
    inference_engines = [
        (
            "continuous_in_phase_fixed_second_half",
            "last",
            LogisticCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      mean_prior_coordination_logit=mean_prior_coordination_logit,
                                                      std_prior_coordination_logit=std_prior_coordination_logit,
                                                      std_coordination_logit_drifting=std_coordination_logit_drifting,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      num_particles=num_particles,
                                                      fix_coordination_on_second_half=True)
        ),
        (
            "continuous_anti_phase_fixed_second_half",
            "last",
            LogisticCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      mean_prior_coordination_logit=mean_prior_coordination_logit,
                                                      std_prior_coordination_logit=std_prior_coordination_logit,
                                                      std_coordination_logit_drifting=std_coordination_logit_drifting,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      num_particles=num_particles,
                                                      f=ANTI_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=True)
        ),
        (
            "continuous_either_phase_fixed_second_half",
            "last",
            LogisticCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      mean_prior_coordination_logit=mean_prior_coordination_logit,
                                                      std_prior_coordination_logit=std_prior_coordination_logit,
                                                      std_coordination_logit_drifting=std_coordination_logit_drifting,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      num_particles=num_particles,
                                                      f=EITHER_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=True)
        ),
        (
            "continuous_in_phase_variable",
            "all",
            LogisticCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      mean_prior_coordination_logit=mean_prior_coordination_logit,
                                                      std_prior_coordination_logit=std_prior_coordination_logit,
                                                      std_coordination_logit_drifting=std_coordination_logit_drifting,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      num_particles=num_particles,
                                                      fix_coordination_on_second_half=False)
        ),
        (
            "continuous_anti_phase_variable",
            "all",
            LogisticCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      mean_prior_coordination_logit=mean_prior_coordination_logit,
                                                      std_prior_coordination_logit=std_prior_coordination_logit,
                                                      std_coordination_logit_drifting=std_coordination_logit_drifting,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      num_particles=num_particles,
                                                      f=ANTI_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=False)
        ),
        (
            "continuous_either_phase_variable",
            "all",
            LogisticCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                      mean_prior_coordination_logit=mean_prior_coordination_logit,
                                                      std_prior_coordination_logit=std_prior_coordination_logit,
                                                      std_coordination_logit_drifting=std_coordination_logit_drifting,
                                                      mean_prior_vocalics=mean_prior_vocalics,
                                                      std_prior_vocalics=std_prior_vocalics,
                                                      std_coordinated_vocalics=std_coordinated_vocalics,
                                                      num_particles=num_particles,
                                                      f=EITHER_PHASE_FUNCTION,
                                                      fix_coordination_on_second_half=False)
        )
    ]

    return inference_engines


def get_truncated_gaussian_inference_mixture_engines(vocalic_series: VocalicsSparseSeries,
                                                     mean_prior_coordination: float,
                                                     std_prior_coordination: float,
                                                     std_coordination_drifting: float,
                                                     mean_prior_vocalics: np.ndarray, std_prior_vocalics: np.ndarray,
                                                     std_uncoordinated_vocalics: np.ndarray,
                                                     std_coordinated_vocalics: np.ndarray, num_particles: int):
    inference_engines = [
        (
            "continuous_in_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationMixtureInference(vocalic_series=vocalic_series,
                                                          mean_prior_coordination=mean_prior_coordination,
                                                          std_prior_coordination=std_prior_coordination,
                                                          std_coordination_drifting=std_coordination_drifting,
                                                          mean_prior_vocalics=mean_prior_vocalics,
                                                          std_prior_vocalics=std_prior_vocalics,
                                                          std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                          std_coordinated_vocalics=std_coordinated_vocalics,
                                                          num_particles=num_particles,
                                                          fix_coordination_on_second_half=True)
        ),
        (
            "continuous_anti_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationMixtureInference(vocalic_series=vocalic_series,
                                                          mean_prior_coordination=mean_prior_coordination,
                                                          std_prior_coordination=std_prior_coordination,
                                                          std_coordination_drifting=std_coordination_drifting,
                                                          mean_prior_vocalics=mean_prior_vocalics,
                                                          std_prior_vocalics=std_prior_vocalics,
                                                          std_coordinated_vocalics=std_coordinated_vocalics,
                                                          std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                          num_particles=num_particles,
                                                          f=ANTI_PHASE_FUNCTION,
                                                          fix_coordination_on_second_half=True)
        ),
        (
            "continuous_either_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationMixtureInference(vocalic_series=vocalic_series,
                                                          mean_prior_coordination=mean_prior_coordination,
                                                          std_prior_coordination=std_prior_coordination,
                                                          std_coordination_drifting=std_coordination_drifting,
                                                          mean_prior_vocalics=mean_prior_vocalics,
                                                          std_prior_vocalics=std_prior_vocalics,
                                                          std_coordinated_vocalics=std_coordinated_vocalics,
                                                          std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                          num_particles=num_particles,
                                                          f=EITHER_PHASE_FUNCTION,
                                                          fix_coordination_on_second_half=True)
        ),
        (
            "continuous_in_phase_variable",
            "all",
            TruncatedGaussianCoordinationMixtureInference(vocalic_series=vocalic_series,
                                                          mean_prior_coordination=mean_prior_coordination,
                                                          std_prior_coordination=std_prior_coordination,
                                                          std_coordination_drifting=std_coordination_drifting,
                                                          mean_prior_vocalics=mean_prior_vocalics,
                                                          std_prior_vocalics=std_prior_vocalics,
                                                          std_coordinated_vocalics=std_coordinated_vocalics,
                                                          std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                          num_particles=num_particles,
                                                          fix_coordination_on_second_half=False)
        ),
        (
            "continuous_anti_phase_variable",
            "all",
            TruncatedGaussianCoordinationMixtureInference(vocalic_series=vocalic_series,
                                                          mean_prior_coordination=mean_prior_coordination,
                                                          std_prior_coordination=std_prior_coordination,
                                                          std_coordination_drifting=std_coordination_drifting,
                                                          mean_prior_vocalics=mean_prior_vocalics,
                                                          std_prior_vocalics=std_prior_vocalics,
                                                          std_coordinated_vocalics=std_coordinated_vocalics,
                                                          std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                          num_particles=num_particles,
                                                          f=ANTI_PHASE_FUNCTION,
                                                          fix_coordination_on_second_half=False)
        ),
        (
            "continuous_either_phase_variable",
            "all",
            TruncatedGaussianCoordinationMixtureInference(vocalic_series=vocalic_series,
                                                          mean_prior_coordination=mean_prior_coordination,
                                                          std_prior_coordination=std_prior_coordination,
                                                          std_coordination_drifting=std_coordination_drifting,
                                                          mean_prior_vocalics=mean_prior_vocalics,
                                                          std_prior_vocalics=std_prior_vocalics,
                                                          std_coordinated_vocalics=std_coordinated_vocalics,
                                                          std_uncoordinated_vocalics=std_uncoordinated_vocalics,
                                                          num_particles=num_particles,
                                                          f=EITHER_PHASE_FUNCTION,
                                                          fix_coordination_on_second_half=False)
        )
    ]

    return inference_engines


def get_truncated_gaussian_blending_latent_vocalics_inference_engines(vocalic_series: VocalicsSparseSeries,
                                                                      mean_prior_coordination: float,
                                                                      std_prior_coordination: float,
                                                                      std_coordination_drifting: float,
                                                                      mean_prior_latent_vocalics: np.ndarray,
                                                                      std_prior_latent_vocalics: np.ndarray,
                                                                      std_coordinated_latent_vocalics: np.ndarray,
                                                                      std_observed_vocalics: np.ndarray,
                                                                      num_particles: int):
    inference_engines = [
        (
            "continuous_in_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=vocalic_series,
                                                                         mean_prior_coordination=mean_prior_coordination,
                                                                         std_prior_coordination=std_prior_coordination,
                                                                         std_coordination_drifting=std_coordination_drifting,
                                                                         mean_prior_latent_vocalics=mean_prior_latent_vocalics,
                                                                         std_prior_latent_vocalics=std_prior_latent_vocalics,
                                                                         std_coordinated_latent_vocalics=std_coordinated_latent_vocalics,
                                                                         std_observed_vocalics=std_observed_vocalics,
                                                                         fix_coordination_on_second_half=True,
                                                                         num_particles=num_particles)
        ),
        (
            "continuous_anti_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=vocalic_series,
                                                                         mean_prior_coordination=mean_prior_coordination,
                                                                         std_prior_coordination=std_prior_coordination,
                                                                         std_coordination_drifting=std_coordination_drifting,
                                                                         mean_prior_latent_vocalics=mean_prior_latent_vocalics,
                                                                         std_prior_latent_vocalics=std_prior_latent_vocalics,
                                                                         std_coordinated_latent_vocalics=std_coordinated_latent_vocalics,
                                                                         std_observed_vocalics=std_observed_vocalics,
                                                                         f=ANTI_PHASE_FUNCTION,
                                                                         fix_coordination_on_second_half=True,
                                                                         num_particles=num_particles)
        ),
        (
            "continuous_either_phase_fixed_second_half",
            "last",
            TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=vocalic_series,
                                                                         mean_prior_coordination=mean_prior_coordination,
                                                                         std_prior_coordination=std_prior_coordination,
                                                                         std_coordination_drifting=std_coordination_drifting,
                                                                         mean_prior_latent_vocalics=mean_prior_latent_vocalics,
                                                                         std_prior_latent_vocalics=std_prior_latent_vocalics,
                                                                         std_coordinated_latent_vocalics=std_coordinated_latent_vocalics,
                                                                         std_observed_vocalics=std_observed_vocalics,
                                                                         f=EITHER_PHASE_FUNCTION,
                                                                         fix_coordination_on_second_half=True,
                                                                         num_particles=num_particles)
        ),
        (
            "continuous_in_phase_variable",
            "all",
            TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=vocalic_series,
                                                                         mean_prior_coordination=mean_prior_coordination,
                                                                         std_prior_coordination=std_prior_coordination,
                                                                         std_coordination_drifting=std_coordination_drifting,
                                                                         mean_prior_latent_vocalics=mean_prior_latent_vocalics,
                                                                         std_prior_latent_vocalics=std_prior_latent_vocalics,
                                                                         std_coordinated_latent_vocalics=std_coordinated_latent_vocalics,
                                                                         std_observed_vocalics=std_observed_vocalics,
                                                                         fix_coordination_on_second_half=False,
                                                                         num_particles=num_particles)
        ),
        (
            "continuous_anti_phase_variable",
            "all",
            TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=vocalic_series,
                                                                         mean_prior_coordination=mean_prior_coordination,
                                                                         std_prior_coordination=std_prior_coordination,
                                                                         std_coordination_drifting=std_coordination_drifting,
                                                                         mean_prior_latent_vocalics=mean_prior_latent_vocalics,
                                                                         std_prior_latent_vocalics=std_prior_latent_vocalics,
                                                                         std_coordinated_latent_vocalics=std_coordinated_latent_vocalics,
                                                                         std_observed_vocalics=std_observed_vocalics,
                                                                         f=ANTI_PHASE_FUNCTION,
                                                                         fix_coordination_on_second_half=False,
                                                                         num_particles=num_particles)
        ),
        (
            "continuous_either_phase_variable",
            "all",
            TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=vocalic_series,
                                                                         mean_prior_coordination=mean_prior_coordination,
                                                                         std_prior_coordination=std_prior_coordination,
                                                                         std_coordination_drifting=std_coordination_drifting,
                                                                         mean_prior_latent_vocalics=mean_prior_latent_vocalics,
                                                                         std_prior_latent_vocalics=std_prior_latent_vocalics,
                                                                         std_coordinated_latent_vocalics=std_coordinated_latent_vocalics,
                                                                         std_observed_vocalics=std_observed_vocalics,
                                                                         f=EITHER_PHASE_FUNCTION,
                                                                         fix_coordination_on_second_half=False,
                                                                         num_particles=num_particles)
        )
    ]

    return inference_engines


def estimate(trials_dir: str, data_dir: str, plot_coordination: bool, model: str, p_prior_coordination,
             p_coordination_transition, mean_prior_coordination: float, std_prior_coordination: float,
             std_coordination_drifting: float, mean_prior_vocalics: float, std_prior_vocalics: float,
             std_uncoordinated_vocalics: float, std_coordinated_vocalics: float, std_observed_vocalics: float,
             num_particles: int, seed: Optional[int]):
    logs_dir = f"{data_dir}/logs"
    plots_dir = f"{data_dir}/plots"

    os.makedirs(logs_dir, exist_ok=True)
    if plot_coordination:
        os.makedirs(plots_dir, exist_ok=True)

    # Constants
    NUM_TIME_STEPS = 17 * 60  # (17 minutes of mission in seconds)
    NUM_FEATURES = 2  # Pitch and Intensity

    mean_prior_vocalics = np.ones(NUM_FEATURES) * mean_prior_vocalics
    std_prior_vocalics = np.ones(NUM_FEATURES) * std_prior_vocalics
    std_uncoordinated_vocalics = np.ones(NUM_FEATURES) * std_uncoordinated_vocalics
    std_coordinated_vocalics = np.ones(NUM_FEATURES) * std_coordinated_vocalics
    std_observed_vocalics = np.ones(NUM_FEATURES) * std_observed_vocalics

    data_table = {
        "trial": [],
        "estimation_name": [],
        "aggregation": [],
        "mean": [],
        "variance": [],
        "means": [],
        "variances": [],
        "score": []
    }

    if not os.path.exists(trials_dir):
        raise Exception(f"Directory {trials_dir} does not exist.")

    filepaths = list(glob(f"{trials_dir}/T*"))
    pbar = tqdm(total=len(filepaths), desc="Inferring Coordination...")
    for i, filepath in enumerate(filepaths):
        if os.path.isdir(filepath):
            trial = Trial.from_directory(filepath)

            configure_log(True, f"{logs_dir}/{trial.metadata.number}")
            logging.getLogger().setLevel(logging.INFO)
            vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics,
                                                                 segmentation_method=SegmentationMethod.KEEP_ALL)

            vocalic_series = vocalics_component.sparse_series(NUM_TIME_STEPS, trial.metadata.mission_start)
            vocalic_series.normalize_per_subject()

            if model == "discrete":
                inference_engines = get_discrete_inference_engines(vocalic_series, p_prior_coordination,
                                                                   p_coordination_transition,
                                                                   mean_prior_vocalics, std_prior_vocalics,
                                                                   std_uncoordinated_vocalics,
                                                                   std_coordinated_vocalics)

            elif model == "truncated_gaussian_blending":
                inference_engines = get_truncated_gaussian_blending_inference_engines(vocalic_series,
                                                                                      mean_prior_coordination,
                                                                                      std_prior_coordination,
                                                                                      std_coordination_drifting,
                                                                                      mean_prior_vocalics,
                                                                                      std_prior_vocalics,
                                                                                      std_coordinated_vocalics)
            elif model == "logistic":
                # mean_prior_coordination and std_coordination_drifting should in reality be mean and std of the
                # coordination logits, before sigmoid function application.
                inference_engines = get_logistic_inference_engines(vocalic_series, mean_prior_coordination,
                                                                   std_prior_coordination, std_coordination_drifting,
                                                                   mean_prior_vocalics, std_prior_vocalics,
                                                                   std_coordinated_vocalics, num_particles)

            elif model == "truncated_gaussian_mixture":
                inference_engines = get_truncated_gaussian_inference_mixture_engines(vocalic_series,
                                                                                     mean_prior_coordination,
                                                                                     std_prior_coordination,
                                                                                     std_coordination_drifting,
                                                                                     mean_prior_vocalics,
                                                                                     std_prior_vocalics,
                                                                                     std_uncoordinated_vocalics,
                                                                                     std_coordinated_vocalics,
                                                                                     num_particles)

            elif model == "truncated_gaussian_blending_latent_vocalics":
                inference_engines = get_truncated_gaussian_blending_latent_vocalics_inference_engines(vocalic_series,
                                                                                                      mean_prior_coordination,
                                                                                                      std_prior_coordination,
                                                                                                      std_coordination_drifting,
                                                                                                      mean_prior_vocalics,
                                                                                                      std_prior_vocalics,
                                                                                                      std_coordinated_vocalics,
                                                                                                      std_observed_vocalics,
                                                                                                      num_particles)
            else:
                raise Exception(f"Invalid model {model}.")

            for estimation_name, aggregation, inference_engine in inference_engines:
                if seed is not None:
                    np.random.seed(seed)
                    random.seed(seed)

                params = inference_engine.estimate_means_and_variances()
                means = params[0]
                variances = params[1]

                mean = means[-1] if aggregation == "last" else np.mean(means)
                variance = variances[-1] if aggregation == "last" else np.var(means)

                data_table["trial"].append(trial.metadata.number)
                data_table["estimation_name"].append(estimation_name)
                data_table["aggregation"].append(aggregation)
                data_table["mean"].append(mean)
                data_table["variance"].append(variance)
                data_table["means"].append(means)
                data_table["variances"].append(variances)
                data_table["score"].append(trial.metadata.team_score)

                if plot_coordination:
                    filepath = f"{plots_dir}/{estimation_name}/{trial.metadata.number}.png"
                    plot(means, np.sqrt(variances), vocalic_series.mask, filepath)

            pbar.update()

    df = pd.DataFrame(data_table)
    df.to_pickle(f"{data_dir}/score_regression_data.pkl")


def plot(means: np.ndarray, stds: np.ndarray, masks: List[int], filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig = plt.figure(figsize=(20, 6))
    plt.plot(range(len(means)), means, marker="o", color="tab:orange", linestyle="--")
    plt.fill_between(range(len(means)), means - stds, means + stds, color='tab:orange', alpha=0.2)
    times, masks = list(zip(*[(t, mask) for t, mask in enumerate(masks) if mask > 0 and t < len(means)]))
    plt.scatter(times, masks, color="tab:green", marker="+")
    plt.xlabel("Time Steps (seconds)")
    plt.ylabel("Coordination")
    add_discrete_coordination_bar(main_ax=fig.gca(),
                                  coordination_series=[np.where(means > 0.5, 1, 0)],
                                  coordination_colors=["tab:orange"],
                                  labels=["Coordination"])

    plt.savefig(filepath, format="png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimates coordination and plot estimates from a list of trials."
    )
    parser.add_argument("--trials_dir", type=str, required=True, help="Directory where serialized trials are located.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the data must be saved.")
    parser.add_argument("--plot_coordination", action="store_true", required=False, default=False,
                        help="Whether plots must be generated. If so, they will be saved under data_dir/plots.")
    parser.add_argument("--model", type=str, required=True, help="Inference model: truncated_gaussian or logistic.")
    parser.add_argument("--pc", type=float, required=False, default=0, help="Probability of C_0 = 1")
    parser.add_argument("--pcc", type=float, required=False, default=0.1, help="Probability of C_t != C_{t-1}")
    parser.add_argument("--mc", type=float, required=False, default=0, help="Mean of C_0 or Logit C_0")
    parser.add_argument("--sc", type=float, required=False, default=0.01, help="Std of C_0 or Logit C_0")
    parser.add_argument("--scc", type=float, required=False, default=0.05, help="Std of C or Logit C drifting.")
    parser.add_argument("--mv", type=float, required=False, default=0, help="Mean of vocalics.")
    parser.add_argument("--sv", type=float, required=False, default=1, help="Std of vocalics.")
    parser.add_argument("--suv", type=int, required=False, default=1, help="Std of uncoupled vocalics.")
    parser.add_argument("--scv", type=int, required=False, default=1, help="Std of coupled vocalics.")
    parser.add_argument("--sov", type=int, required=False, default=0.5, help="Std of observed vocalics.")
    parser.add_argument("--num_particles", type=int, required=False, default=10000,
                        help="Number of particles for the particle based model.")
    parser.add_argument("--seed", type=int, required=False, help="Seed for the particle based model.")

    args = parser.parse_args()

    estimate(args.trials_dir, args.data_dir, args.plot_coordination, args.model, args.pc, args.pcc, args.mc, args.sc,
             args.scc, args.mv, args.sv, args.suv, args.scv, args.sov, args.num_particles, args.seed)

    # Save hyper-parameters
    with open(f"{args.data_dir}/hyper_params.txt", "w") as f:
        json.dump(vars(args), f, indent=4)
