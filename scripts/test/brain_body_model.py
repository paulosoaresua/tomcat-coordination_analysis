# from typing import Optional
#
# from copy import copy
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# from coordination.common.log import BaseLogger, TensorBoardLogger
# from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
# from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
#     BetaCoordinationLatentVocalicsTrainingHyperParameters

from coordination.model.brain_body_model import BrainBodyModel

# Parameters
TIME_STEPS = 50
NUM_SERIES = 2
NUM_CHANNELS = 2
SEED = 0

if __name__ == "__main__":
    model = BrainBodyModel(0.1, NUM_CHANNELS, 3)

    model.sd_uc = 0.1
    model.sd_brain = 1
    model.sd_body = 1
    model.sd_obs_brain = 1
    model.sd_obs_body = 1

    model.sample(NUM_SERIES, TIME_STEPS, SEED)

