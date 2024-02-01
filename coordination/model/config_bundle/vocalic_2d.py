from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np

from coordination.common.normalization import NORMALIZATION_PER_FEATURE
from coordination.model.config_bundle.bundle import ModelConfigBundle


@dataclass
class Vocalic2DConfigBundle(ModelConfigBundle):
    """
    Container for the different parameters of the vocalic model.
    """

    num_subjects: int = 3
    num_time_steps_in_coordination_scale: int = 100

    # When the following is true, we will estimate coordination only at the time steps for which we
    # have observed vocalic features.
    match_vocalic_scale: bool = True

    # Used for both sampling and inference. It is capped to the number of time steps in
    # coordination scale.
    num_time_steps_to_fit: int = None

    # Fixed value of coordination during inference
    observed_coordination_for_inference: Union[float, np.ndarray] = None

    # Initial sampled values of coordination. In a call to draw samples, the model will start by
    # using the samples given here and move forward sampling in time until the number of time
    # steps desired have been reached.
    initial_coordination_samples: np.ndarray = None
    initial_state_space_samples: np.ndarray = None

    # Whether to use a constant model of coordination
    constant_coordination: bool = False

    # Parameters for inference
    mean_mean_uc0: float = 0.0  # Variable coordination
    sd_mean_uc0: float = 5.0
    sd_sd_uc: float = 1.0
    alpha_c: float = 1.0  # Fix coordination
    beta_c: float = 1.0
    # -----------------------
    # State space and observation
    mean_mean_a0: float = 0.0
    sd_mean_a0: float = 1.0
    sd_sd_a: float = 1.0
    sd_sd_o: float = 1.0

    # Given parameter values. Required for sampling, not for inference.
    mean_uc0: float = 0.0  # Coordination = 0.5
    sd_uc: float = 0.5
    mean_a0: float = None
    sd_a: float = None
    sd_o: float = 0.1

    # Sampling settings
    sampling_time_scale_density: float = 1.0
    allow_sampled_subject_repetition: bool = False
    fix_sampled_subject_sequence: bool = True

    # Inference settings
    observation_normalization: str = NORMALIZATION_PER_FEATURE

    # Modules settings
    num_vocalic_features: int = 4
    vocalic_feature_names: List[str] = field(
        default_factory=lambda: ["pitch", "intensity", "jitter", "shimmer"]
    )

    share_mean_a0_across_subjects: bool = False
    share_mean_a0_across_dimensions: bool = False
    share_sd_a_across_subjects: bool = True
    share_sd_a_across_dimensions: bool = False
    share_sd_o_across_subjects: bool = True
    share_sd_o_across_dimensions: bool = True

    num_hidden_layers: int = 0
    hidden_dimension_size: int = 0
    activation: str = "linear"
    # Only position is used. # From position to 4 vocalic features
    weights: List[np.ndarray] = field(default_factory=lambda: [np.ones((1, 4))])
    mean_w0: float = 0.0
    sd_w0: float = 1.0

    # To allow splitting features into different groups
    # If provided, it must be list of a dictionaries in the following format:
    # {
    # "name": "name of the group"
    # "features": ["a list vocalic features to include in this group"]
    # "weights": None or fixed weights to transform the latent component of the group to
    # observations. If not given, it will be fit to the data.
    # Example:
    # vocalic_groups = [
    #     {
    #         "name": "pitch_intensity",
    #         "features": ["pitch", "intensity"],
    #         "weights": [np.ones((1, 2))]
    #     },
    #     {
    #         "name": "jitter_shimmer",
    #         "features": ["jitter", "shimmer"],
    #         "weights": [np.ones((1, 2))]
    #     }
    # ]
    vocalic_groups: List[Dict[str, Any]] = None

    # Metadata parameters. These must be filled before inference.
    time_steps_in_coordination_scale: np.ndarray = None
    subject_indices: np.ndarray = None
    prev_time_same_subject: np.ndarray = None
    prev_time_diff_subject: np.ndarray = None
    observed_values: np.ndarray = None


class Vocalic2DSemanticLinkConfigBundle(Vocalic2DConfigBundle):
    """
    Container for the different parameters of the vocalic2D + semantic link model.
    """

    a_p: float = 1.0
    b_p: float = 1.0
    p: float = None

    # Metadata parameters. These must be filled before inference.
    semantic_link_time_steps_in_coordination_scale: np.ndarray = None
