import matplotlib.pyplot as plt
import numpy as np

# Custom code
from coordination.common.dataset import Dataset, SeriesData
from coordination.component.speech.common import VocalicsSparseSeries
from coordination.inference.beta_coordination import BetaCoordinationInferenceFromVocalics
from coordination.plot.coordination import plot_continuous_coordination, add_continuous_coordination_bar
from coordination.plot.vocalics import plot_vocalic_features
from coordination.synthetic.coordination.beta_coordination_generator import BetaCoordinationGenerator
from coordination.synthetic.component.speech.continuous_coordination_vocalics_blending_generator import \
    ContinuousCoordinationVocalicsBlendingGenerator

if __name__ == "__main__":
    # Constants
    SEED = 0  # For reproducibility
    OBSERVATION_DENSITY = 1  # Proportion of timesteps with observation
    NUM_TIME_STEPS = 100
    M = int(NUM_TIME_STEPS / 2)  # We assume coordination in the second half of the period is constant for now
    NUM_FEATURES = 2

    # Parameters of the distributions
    A0 = 1E-16;
    B0 = 1E16  # The process starts with no coordination
    STD_COORDINATION_DRIFT = 0.1  # Coordination drifts by a little
    MEAN_PRIOR_VOCALICS = np.zeros(NUM_FEATURES)
    STD_PRIOR_VOCALICS = np.ones(NUM_FEATURES)
    STD_COORDINATED_VOCALICS = np.ones(NUM_FEATURES)

    generator = BetaCoordinationGenerator(num_time_steps=NUM_TIME_STEPS,
                                          prior_a=A0,
                                          prior_b=B0,
                                          transition_std=STD_COORDINATION_DRIFT)
    continuous_cs = generator.generate(SEED)
    continuous_cs[M:] = continuous_cs[M]

    generator = ContinuousCoordinationVocalicsBlendingGenerator(coordination_series=continuous_cs,
                                                                num_vocalic_features=NUM_FEATURES,
                                                                time_scale_density=OBSERVATION_DENSITY,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS)
    vocalic_series = generator.generate(SEED)

    inference_engine = BetaCoordinationInferenceFromVocalics(prior_a=A0,
                                                             prior_b=B0,
                                                             std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                             mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                             std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                             std_coordinated_vocalics=STD_COORDINATED_VOCALICS)

    dataset = Dataset([SeriesData(vocalic_series), SeriesData(vocalic_series)])
    np.random.seed(SEED)

    for params in inference_engine.predict(dataset, 10000):
        mean_cs = params[0]
        var_cs = params[1]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(M + 1), mean_cs, marker="o", color="tab:orange", linestyle="--", label='Estimated')
        plt.fill_between(range(M + 1), mean_cs - np.sqrt(var_cs), mean_cs + np.sqrt(var_cs), color='tab:orange', alpha=0.2)
        plt.plot(range(M + 1), continuous_cs[0:M + 1], marker="o", color="tab:blue", linestyle="--", label='Ground Truth')
        plt.xlabel("Time Steps (seconds)")
        plt.ylabel("Coordination")
        plt.title("Continuous Coordination Inference", fontsize=14, weight="bold")
        plt.legend()
        plt.show()


