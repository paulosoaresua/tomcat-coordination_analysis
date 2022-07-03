from src.trial import Trial
from src.components.speech.vocalics_aggregator import VocalicsAggregator
from src.components.speech.vocalics_writer import VocalicsWriter

if __name__ == "__main__":
    # trial = Trial()
    # trial.parse("../data/asist/study2/metadata/.metadata")
    #
    # aggregator = VocalicsAggregator(trial.utterances_per_subject)
    # component = aggregator.split_with_current_utterance_truncation()
    # VocalicsWriter().write("../data/asist/study2/evidence", component, trial.start, 1/1000)

    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from scipy.stats import norm, multivariate_normal
    from typing import Any, Dict, List

    # Custom code
    from src.synthetic.coordination_generator import ContinuousCoordinationGenerator
    from src.synthetic.vocalics_generator import VocalicsGeneratorForContinuousCoordination
    from src.transformations.series_transformations import embed_features_across_dimensions
    from src.inference.vocalics import estimate_continuous_coordination

    SEED = 0  # For reproducibility
    MEAN_SHIFT = 0  # Prior series A/B mean
    OBSERVATION_DENSITY = 1  # Inference is harder if density is small

    random.seed(SEED)
    np.random.seed(SEED)
    num_time_steps = 100
    continuous_cs = ContinuousCoordinationGenerator().generate_evidence(num_time_steps)

    random.seed(SEED)
    np.random.seed(SEED)
    vocalics_generator = VocalicsGeneratorForContinuousCoordination(coordination_series=continuous_cs,
                                                                    vocalic_features=["pitch", "intensity"],
                                                                    time_scale_density=OBSERVATION_DENSITY,
                                                                    mean_shift_coupled=MEAN_SHIFT,
                                                                    var_coupled=1)
    series_a, series_b = vocalics_generator.generate_evidence()


    def pdf_ab(value: np.ndarray, previous_same: Any, previous_other: Any, coordination: float):
        pdf = (multivariate_normal.pdf(value) * (1 - coordination) +
               multivariate_normal.pdf(value, mean=MEAN_SHIFT + previous_other) * coordination)
        return np.prod(pdf)

    multi_dim_a, mask_a = embed_features_across_dimensions(num_time_steps, series_a)
    multi_dim_b, mask_b = embed_features_across_dimensions(num_time_steps, series_b)
    samples = estimate_continuous_coordination(1000, multi_dim_a, multi_dim_b, 0, 0, mask_a, mask_b)

    print(np.sqrt(np.sum(np.square((np.mean(samples[-200:, :], axis=0) - continuous_cs)))))

