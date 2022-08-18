import numpy as np
import pickle
from src.inference.vocalics import DiscreteCoordinationInference
from src.transformations.series_transformations import embed_features_across_dimensions

if __name__ == "__main__":
    with open("../data/asist/study3/series/T000745/vocalics/series_a.pkl", "rb") as f:
        series_a = pickle.load(f)
    with open("../data/asist/study3/series/T000745/vocalics/series_b.pkl", "rb") as f:
        series_b = pickle.load(f)

    # Constants
    T = 100
    M = int(T / 2)
    PC = 0.9  # Coordination is likely to be preserved from one time step to the next

    multi_dim_a, mask_a = embed_features_across_dimensions(1020, series_a)
    multi_dim_b, mask_b = embed_features_across_dimensions(1020, series_b)

    # Normalize series to have mean 0 and std 1
    multi_dim_a = (multi_dim_a - np.mean(multi_dim_a, axis=0)) / np.std(multi_dim_a, axis=0)
    multi_dim_b = (multi_dim_b - np.mean(multi_dim_b, axis=0)) / np.std(multi_dim_b, axis=0)

    inference_engine = DiscreteCoordinationInference(multi_dim_a.T, multi_dim_b.T, 0, PC, 0, 1, 0, 1, 1,
                                                     np.array(mask_a), np.array(mask_b))
    marginals = inference_engine.estimate_marginals()
