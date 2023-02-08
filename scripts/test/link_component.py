import arviz as az
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

from coordination.model.components.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.model.components.link_component import LinkComponent

from coordination.common.functions import sigmoid

if __name__ == "__main__":
    T = 100

    # Components
    coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination=0.5, sd_uc=1)
    link_cpn = LinkComponent("link", a_p=1, b_p=1)

    # Parameters
    coordination_cpn.parameters.sd_uc.value = np.full(1, fill_value=0.1)
    link_cpn.parameters.p.value = 0.5

    # Samples
    coordination_values = coordination_cpn.draw_samples(num_series=1, num_time_steps=T, seed=0).coordination
    link_samples = link_cpn.draw_samples(num_series=1, seed=None, coordination=coordination_values)

    # Plot coordination
    plt.figure()
    plt.plot(np.arange(T), coordination_values[0], marker="o", color="tab:blue", markersize=5)
    plt.title(f"Samples Coordination")
    plt.show()

    # Joint inference
    link_cpn.parameters.clear_values()
    coordination_cpn.parameters.clear_values()
    with pm.Model(coords={"coord_time": np.arange(T)}) as model:
        _, coordination = coordination_cpn.update_pymc_model("coord_time")
        link_cpn.update_pymc_model(
            coordination=coordination[link_samples.time_steps_in_coordination_scale[0]],
            observation=np.ones(link_samples.num_time_steps))

        # Predictive prior
        idata = pm.sample_prior_predictive(random_seed=0)

        prior_obs = idata.prior_predictive["link_link"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(link_samples.num_time_steps)[:, None].repeat(500, axis=1), prior_obs.T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(link_samples.num_time_steps), np.ones(link_samples.num_time_steps), marker="o",
                 color="black", markersize=5)
        plt.title(f"Prior Predictive Link")
        plt.show()

        # Posterior
        idata = pm.sample(draws=1000, tune=1000, chains=2, init="jitter+adapt_diag", random_seed=0)

        #   Parameters
        az.plot_trace(idata, var_names=["p_link", "sd_uc"])
        plt.show()

        #   Coordination
        posterior = sigmoid(idata.posterior["unbounded_coordination"].sel(chain=0).to_numpy())

        plt.figure()
        plt.plot(np.arange(T)[:, None].repeat(1000, axis=1), posterior.T, color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), coordination_values[0], marker="o", color="black", alpha=1, markersize=5)
        plt.scatter(link_samples.time_steps_in_coordination_scale[0],
                    coordination_values[0, link_samples.time_steps_in_coordination_scale[0]], c="tab:red",
                    marker="*", s=5, zorder=3)
        plt.plot(np.arange(T), posterior.mean(axis=0), color="tab:pink", alpha=1)
        plt.title(f"Posterior Coordination")
        plt.show()
