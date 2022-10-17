
import pickle

import numpy as np
import os

from coordination.common.dataset import InputFeaturesDataset

agents = ["no_advisor", "tomcat_agent", "other_agents"]

for i in [1, 2]:
    all_features = []
    all_scores = []
    all_estimates = []

    for agent in agents:
        root_dir = f"/Users/paulosoares/code/tomcat-coordination/data/study-3_2022/{agent}/estimates/gaussian_latent"
        dataset_path = f"{root_dir}/mission{i}/coordination/estimation/dataset.pkl"
        estimation_path = f"{root_dir}/mission{i}/coordination/estimation/estimates.pkl"

        with open(dataset_path, "rb") as f:
            input_features, scores = pickle.load(f)
            all_features.append(input_features)
            all_scores.append(scores)

        with open(estimation_path, "rb") as f:
            estimates = pickle.load(f)
            all_estimates.extend(estimates)

    mission_features = InputFeaturesDataset.merge_list(all_features)
    mission_scores = np.concatenate(all_scores)

    mission_dataset = (mission_features, mission_scores)
    mission_estimates = all_estimates

    root_dir = f"/Users/paulosoares/code/tomcat-coordination/data/study-3_2022/all_agents/estimates/gaussian_latent"
    path = f"{root_dir}/mission{i}/coordination/estimation/estimates.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mission_estimates, f)

    # Save the original dataset used to estimate coordination
    path = f"{root_dir}/mission{i}/coordination/estimation/dataset.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(mission_dataset, f)
