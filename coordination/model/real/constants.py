import numpy as np

from coordination.common.constants import DEFAULT_NUM_SUBJECTS


class VocalicConstants:
    """
    This class contains default values for a vocalics group.
    """

    NUM_VOCALIC_FEATURES = 4
    VOCALIC_FEATURE_NAMES = ["pitch", "intensity", "jitter", "shimmer"]
    STATE_SPACE_DIM_SIZE = NUM_VOCALIC_FEATURES
    STATE_SPACE_DIM_NAMES = VOCALIC_FEATURE_NAMES
    SELF_DEPENDENT_STATE_SPACE = True
    DEFAULT_OBSERVATION_NORMALIZATION = True

    # Sharing options
    SHARE_MEAN_A0_ACROSS_SUBJECT = False
    SHARE_MEAN_A0_ACROSS_DIMENSIONS = False
    SHARE_SD_A_ACROSS_SUBJECTS = True
    SHARE_SD_A_ACROSS_DIMENSIONS = False
    SHARE_SD_O_ACROSS_SUBJECTS = True
    SHARE_SD_O_ACROSS_DIMENSIONS = True

    # For inference
    SD_MEAN_UC0 = 5
    SD_SD_UC = 1
    MEAN_MEAN_A0 = np.zeros((DEFAULT_NUM_SUBJECTS, STATE_SPACE_DIM_SIZE))
    SD_MEAN_A0 = np.ones((DEFAULT_NUM_SUBJECTS, STATE_SPACE_DIM_SIZE))
    # Same variance across subjects but not dimensions
    SD_SD_A = np.ones(STATE_SPACE_DIM_SIZE)
    # Same variance across subjects and dimensions
    SD_SD_O = np.ones(1)

    # For sample generation
    MEAN_UC0 = 0
    SD_UC = 0.5  # this is fixed during inference as well
    MEAN_A0 = np.zeros_like(MEAN_MEAN_A0)
    SD_A = np.ones_like(SD_SD_A) * 0.1
    SD_O = np.ones_like(SD_SD_O) * 0.1  # this is fixed during inference as well

    SAMPLING_TIME_SCALE_DENSITY = 1.0
    ALLOW_SAMPLED_SUBJECT_REPETITION = False
    FIX_SAMPLED_SUBJECT_SEQUENCE = True

    # Transformations.
    # With the values below, if the dimension of the state space is smaller than the dimension of
    # the observations, the MLP will learn a single matrix of values that transforms the state
    # space vector to a vector with the dimensions in the observed space by linear combination.
    # If the dimension of the state space is 1, the matrix becomes a vector that can be
    # interpreted as the mean of the observations which are scaled by the value in the state space.
    NUM_HIDDEN_LAYERS = 0
    HIDDEN_DIMENSION_SIZE = 0
    ACTIVATION = "linear"
    WEIGHTS = np.ones(
        (NUM_HIDDEN_LAYERS + 1, STATE_SPACE_DIM_SIZE, NUM_VOCALIC_FEATURES)
    )
    MEAN_W0 = 0
    SD_W0 = 1


class SemanticLinkConstants:
    A_P = 1
    B_P = 1
    P = 1
