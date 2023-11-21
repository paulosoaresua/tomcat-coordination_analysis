import numpy as np

DEFAULT_NUM_TIME_STEPS = 100
DEFAULT_NUM_SUBJECTS = 3
SUBJECT_NAMES = ["S1", "S2", "S3"]
BURN_IN = 2000
NUM_SAMPLES = 2000
TARGET_ACCEPT = 0.8


class VocalicConstants:
    """
    This class contains default values for a vocalics group.
    """

    NUM_VOCALIC_FEATURES = 4
    VOCALIC_FEATURE_NAMES = ["pitch", "intensity", "jitter", "shimmer"]
    STATE_SPACE_DIM_SIZE = NUM_VOCALIC_FEATURES
    STATE_SPACE_DIM_NAMES = VOCALIC_FEATURE_NAMES
    SELF_DEPENDENT_STATE_SPACE = True

    # Sharing options
    SHARE_MEAN_A0_ACROSS_SUBJECT = False
    SHARE_MEAN_A0_ACROSS_DIMENSIONS = False
    SHARE_SD_A_ACROSS_SUBJECTS = True
    SHARE_SD_A_ACROSS_DIMENSIONS = False
    SHARE_SD_O_ACROSS_SUBJECTS = True
    SHARE_SD_O_ACROSS_DIMENSIONS = True

    # For inference
    SD_MEAN_UC0 = 1
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
