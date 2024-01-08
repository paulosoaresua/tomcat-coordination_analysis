from dataclasses import dataclass, field
from typing import List

import numpy as np

from coordination.common.normalization import (
    NORMALIZATION_PER_FEATURE, NORMALIZATION_PER_SUBJECT_AND_FEATURE)
from coordination.model.config_bundle.bundle import ModelConfigBundle


@dataclass
class VocalicSemanticLinkConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the vocalic + semantic link model.
    """

    num_subjects: int = 3
    num_time_steps_in_coordination_scale: int = 100
    observation_normalization: str = NORMALIZATION_PER_SUBJECT_AND_FEATURE
    state_space_dimension_size: int = 4
    state_space_dimension_names: List[str] = field(
        default_factory=lambda: ["pitch", "intensity", "jitter", "shimmer"]
    )
    self_dependent: bool = True
    num_vocalic_features: int = 4
    vocalic_feature_names: List[str] = field(
        default_factory=lambda: ["pitch", "intensity", "jitter", "shimmer"]
    )

    # Hyper priors
    mean_mean_uc0: float = 0.0
    sd_mean_uc0: float = 5.0
    sd_sd_uc: float = 1.0
    mean_mean_a0: float = 0.0
    sd_mean_a0: float = 1.0
    sd_sd_a: float = 1.0
    sd_sd_o: float = 1.0
    a_p: float = 1.0
    b_p: float = 1.0

    share_mean_a0_across_subjects: bool = False
    share_mean_a0_across_dimensions: bool = False
    share_sd_a_across_subjects: bool = True
    share_sd_a_across_dimensions: bool = False
    share_sd_o_across_subjects: bool = True
    share_sd_o_across_dimensions: bool = True

    sampling_time_scale_density: float = 1.0
    allow_sampled_subject_repetition: bool = False
    fix_sampled_subject_sequence: bool = True

    # Some parameters are given and others fixed.
    mean_uc0: float = None
    sd_uc: float = 0.5
    mean_a0: float = None
    sd_a: float = None
    sd_o: float = 0.1
    p: float = None

    # Fixed coordination series for sampling.
    coordination_samples: np.ndarray = None

    # Evidence and metadata filled before inference.
    vocalics_time_steps_in_coordination_scale: np.ndarray = None
    subject_indices: np.ndarray = None
    prev_time_same_subject: np.ndarray = None
    prev_time_diff_subject: np.ndarray = None
    observed_vocalic_values: np.ndarray = None
    semantic_link_time_steps_in_coordination_scale: np.ndarray = None

    # To transform a high-dimension state space to a lower dimension observation in case we
    # want to observe position only.
    num_hidden_layers: int = 0
    hidden_dimension_size: int = 0
    activation: str = "linear"
    weights: List[np.ndarray] = field(default_factory=lambda: [np.eye(4)])
    mean_w0: float = 0.0
    sd_w0: float = 1.0


@dataclass
class Vocalic2DSemanticLinkConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the vocalic 2D + semantic link model.
    """

    num_subjects: int = 3
    num_time_steps_in_coordination_scale: int = 100
    observation_normalization: str = NORMALIZATION_PER_FEATURE
    num_vocalic_features: int = 4
    vocalic_feature_names: List[str] = field(
        default_factory=lambda: ["pitch", "intensity", "jitter", "shimmer"]
    )

    # Hyper priors
    mean_mean_uc0: float = 0.0
    sd_mean_uc0: float = 1.0
    sd_sd_uc: float = 1.0
    mean_mean_a0: float = 0.0
    sd_mean_a0: float = 1.0
    sd_sd_a: float = 1.0
    sd_sd_o: float = 1.0
    a_p: float = 1.0
    b_p: float = 1.0

    share_mean_a0_across_subjects: bool = False
    share_mean_a0_across_dimensions: bool = False
    share_sd_a_across_subjects: bool = True
    share_sd_a_across_dimensions: bool = True
    share_sd_o_across_subjects: bool = True
    share_sd_o_across_dimensions: bool = True

    sampling_time_scale_density: float = 1.0
    allow_sampled_subject_repetition: bool = False
    fix_sampled_subject_sequence: bool = True

    # For sampling. Defaults to None for inference.
    mean_uc0: float = None
    sd_uc: float = None
    mean_a0: float = None
    sd_a: float = None
    sd_o: float = None
    p = None
    # Fixed coordination series for sampling.
    coordination_samples: np.ndarray = None

    # Evidence and metadata filled before inference.
    vocalics_time_steps_in_coordination_scale: np.ndarray = None
    subject_indices: np.ndarray = None
    prev_time_same_subject: np.ndarray = None
    prev_time_diff_subject: np.ndarray = None
    observed_vocalic_values: np.ndarray = None
    semantic_link_time_steps_in_coordination_scale: np.ndarray = None

    # To transform a high-dimension state space to a lower dimension observation in case we
    # want to observe position only.
    num_hidden_layers: int = 0
    hidden_dimension_size: int = 0
    activation: str = "linear"
    # Only position is used. # From position to 4 vocalic features
    weights: List[np.ndarray] = field(default_factory=lambda: [np.ones((1, 4))])
    mean_w0: float = 0.0
    sd_w0: float = 1.0
