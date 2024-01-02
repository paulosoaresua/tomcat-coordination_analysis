import numpy as np


class ConversationConstants:
    """
    This class contains default values for a conversation model.
    """

    SUBJECT_NAMES = ["Bob", "Alice", "Dan"]
    NUM_SUBJECTS = 3
    SQUARED_ANGULAR_FREQUENCY = np.array([1, 0.5, 0.1])
    DAMPENING_COEFFICIENT = 0.0
    TIME_STEP_SIZE_IN_SECONDS = 0.2
    OBSERVATION_NORMALIZATION = None
    BLEND_POSITION = True
    BLEND_SPEED = False
    OBSERVATION_DIM_SIZE = 2

    # Sharing options
    SHARE_MEAN_A0_ACROSS_SUBJECT = False
    SHARE_MEAN_A0_ACROSS_DIMENSIONS = False
    SHARE_SD_A_ACROSS_SUBJECTS = True
    SHARE_SD_A_ACROSS_DIMENSIONS = True
    SHARE_SD_O_ACROSS_SUBJECTS = True
    SHARE_SD_O_ACROSS_DIMENSIONS = True

    # For inference
    MEAN_MEAN_UC0 = 0.0
    SD_MEAN_UC0 = 1.0
    SD_SD_UC = 1.0
    MEAN_MEAN_A0 = 0.0
    SD_MEAN_A0 = 1.0
    SD_SD_A = 1.0
    SD_SD_O = 1.0

    # For sample generation
    MEAN_UC0 = 0.0
    SD_UC = 1
    MEAN_A0 = np.array([[1, 0], [1, 0], [1, 0]])
    SD_A = 0.1
    SD_O = 0.01

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
    WEIGHTS = [np.array([[1], [0]])]  # We observe the position and disregard speed
    MEAN_W0 = 0
    SD_W0 = 1

# Common to both models
# MEAN_UC0 = 0
# SD_UC = 1
# SD_MEAN_UC0 = 1
# SD_SD_UC = 1
# SD_SD_A = np.ones(1)
# SD_SD_O = np.ones(1)
# SHARE_MEAN_A0_ACROSS_SUBJECT = False
# SHARE_MEAN_A0_ACROSS_DIMENSIONS = False
# SHARE_SD_ACROSS_SUBJECTS = True  # same variance across subjects
# SHARE_SD_ACROSS_DIMENSIONS = True  # same variance for position and speed
# NUM_TIME_STEPS = 100
#
# BURN_IN = 2000
# NUM_SAMPLES = 2000
# TARGET_ACCEPT = 0.9

# Conversation model
# NUM_SUBJECTS = 3
#
# #   For sample generation
# INITIAL_STATE_CONVERSATION_MODEL = np.array([[1, 0], [1, 0], [1, 0]])
# SD_A_CONVERSATION_MODEL = np.array([0.1])
# SD_O_CONVERSATION_MODEL = np.array([0.01])
# # ---
# SUBJECT_NAMES_CONVERSATION_MODEL = ["Bob", "Alice", "Dan"]
# ANGULAR_FREQUENCIES_CONVERSATION_MODEL = np.array([1, 0.5, 0.1])
# DAMPENING_COEFFICIENTS_CONVERSATION_MODEL = np.zeros(NUM_SUBJECTS)
# DT_CONVERSATION_MODEL = 0.2
# MEAN_MEAN_A0_CONVERSATION_MODEL = np.zeros((NUM_SUBJECTS, 2))
# SD_MEAN_A0_CONVERSATION_MODEL = np.ones((NUM_SUBJECTS, 2)) * max(
#     INITIAL_STATE_CONVERSATION_MODEL[:, 0]
# )
# SAMPLING_TIME_SCALE_DENSITY_CONVERSATIONAL_MODEL = 1.0
# ALLOW_SAMPLED_SUBJECT_REPETITION_CONVERSATIONAL_MODEL = False
# FIX_SAMPLED_SUBJECT_SEQUENCE_CONVERSATIONAL_MODEL = True
#
# # Spring model
# NUM_SPRINGS = 3
#
# #   For sample generation
# INITIAL_STATE_SPRING_MODEL = np.array([[1, 0], [3, 0], [5, 0]])
# SD_A_SPRING_MODEL = np.array([0.1])
# SD_O_SPRING_MODEL = np.array([0.1])
# # ---
# SPRING_NAMES_SPRING_MODEL = ["Spring 1", "Spring 2", "Spring 3"]
# MASS_SPRING_MODEL = np.ones(NUM_SPRINGS) * 10
# SPRING_CONSTANT_SPRING_MODEL = np.array([16, 8, 4])
# DAMPENING_COEFFICIENTS_SPRING_MODEL = np.zeros(NUM_SPRINGS)
# DT_SPRING_MODEL = 1.0
# MEAN_MEAN_A0_SPRING_MODEL = np.zeros((NUM_SPRINGS, 2))
# SD_MEAN_A0_SPRING_MODEL = np.ones((NUM_SPRINGS, 2)) * max(
#     INITIAL_STATE_SPRING_MODEL[:, 0]
# )
# SAMPLING_RELATIVE_FREQUENCY_SPRING_MODEL = 1.0
