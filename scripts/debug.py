import matplotlib.pyplot as plt
import numpy as np

# Custom code
from coordination.component.speech.common import VocalicsSparseSeries
from coordination.inference.truncated_gaussian_coordination_blending_latent_vocalics import \
    TruncatedGaussianCoordinationBlendingInferenceLatentVocalics
from coordination.plot.coordination import plot_continuous_coordination, add_continuous_coordination_bar
from coordination.plot.vocalics import plot_vocalic_features
from coordination.synthetic.coordination.truncated_gaussian_coordination_generator import \
    TruncatedGaussianCoordinationGenerator
from coordination.synthetic.component.speech.continuous_coordination_latent_vocalics_blending_generator import \
    ContinuousCoordinationLatentVocalicsBlendingGenerator

if __name__ == "__main__":
    # Constants
    SEED = 0  # For reproducibility
    OBSERVATION_DENSITY = 1  # Proportion of timesteps with observation
    NUM_TIME_STEPS = 100
    M = int(NUM_TIME_STEPS / 2)  # We assume coordination in the second half of the period is constant for now
    NUM_FEATURES = 2

    # Parameters of the distributions
    MEAN_COORDINATION_PRIOR = 0
    STD_COORDINATION_PRIOR = 1E-16  # The process starts with no coordination
    STD_COORDINATION_DRIFT = 0.1  # Coordination drifts by a little
    MEAN_PRIOR_VOCALICS = np.zeros(NUM_FEATURES)
    STD_PRIOR_VOCALICS = np.ones(NUM_FEATURES)
    STD_COORDINATED_VOCALICS = np.ones(NUM_FEATURES)
    STD_OBSERVED_VOCALICS = np.ones(NUM_FEATURES) * 0.1

    generator = TruncatedGaussianCoordinationGenerator(num_time_steps=NUM_TIME_STEPS,
                                                       mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                       std_prior_coordination=STD_COORDINATION_PRIOR,
                                                       std_coordination_drifting=STD_COORDINATION_DRIFT)
    continuous_cs = generator.generate(SEED)
    continuous_cs[M:] = continuous_cs[M]

    generator = ContinuousCoordinationLatentVocalicsBlendingGenerator(coordination_series=continuous_cs,
                                                                      num_vocalic_features=NUM_FEATURES,
                                                                      time_scale_density=OBSERVATION_DENSITY,
                                                                      mean_prior_latent_vocalics=MEAN_PRIOR_VOCALICS,
                                                                      std_prior_latent_vocalics=STD_PRIOR_VOCALICS,
                                                                      std_coordinated_vocalics=STD_COORDINATED_VOCALICS,
                                                                      std_observed_vocalics=STD_OBSERVED_VOCALICS,
                                                                      num_speakers=2)

    latent_vocalics, observed_vocalics = generator.generate(SEED)

    inference_engine = TruncatedGaussianCoordinationBlendingInferenceLatentVocalics(vocalic_series=observed_vocalics,
                                                                                    mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                                    std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                                    std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                                    mean_prior_latent_vocalics=MEAN_PRIOR_VOCALICS,
                                                                                    std_prior_latent_vocalics=STD_PRIOR_VOCALICS,
                                                                                    std_coordinated_latent_vocalics=STD_COORDINATED_VOCALICS,
                                                                                    std_observed_vocalics=STD_OBSERVED_VOCALICS,
                                                                                    num_particles=10000)

    params = inference_engine.estimate_means_and_variances()
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
