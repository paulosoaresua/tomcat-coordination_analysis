from __future__ import annotations

import os
import sys
from typing import Any, Callable, List, Optional, Tuple, Union

sys.path.append("../")  # So we can use the coordination package

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor.tensor as ptt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from coordination.model.conversation_model import (ConversationModel,
                                                   ConversationSamples,
                                                   ConversationSeries)
from coordination.model.coordination_model import CoordinationPosteriorSamples
from coordination.model.spring_model import SpringModel
from coordination.module.serial_mass_spring_damper_component import logp

# sample = ptt.as_tensor([[1, 2, 3], [4, 5, 6]])
# initial_mean = ptt.as_tensor([[0, 3, 5], [0, 0, 0]])
# sigma = ptt.as_tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
# coordination = ptt.as_tensor([0.2, 0.2, 0.2])
# prev_time_same_subject = ptt.as_tensor([-1, -1, 0])
# prev_time_diff_subject = ptt.as_tensor([-1, 0, 1])
# prev_same_subject_mask = ptt.as_tensor([0, 0, 1])
# prev_diff_subject_mask = ptt.as_tensor([0, 1, 1])
# self_dependent = ptt.as_tensor(1)
# F_inv = ptt.as_tensor(np.random.rand(3, 2, 2).flatten())
# logp(sample, initial_mean, sigma, coordination, prev_time_same_subject, prev_time_diff_subject,
#      prev_same_subject_mask, prev_diff_subject_mask, self_dependent, F_inv)

# plt.style.use("seaborn-v0_8-darkgrid")
sns.set_style("white")
tex_fonts = {
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.titlesize": 8,
    "axes.linewidth": 1,
}
plt.rcParams.update(tex_fonts)
plt.rc("pdf", fonttype=42)
plt.rcParams["text.usetex"] = True

# DOC_WIDTH = 397.48 # NeurIPS
# DOC_WIDTH = 487.82 #AISTATS
DOC_WIDTH = 400  # AISTATS


def calculate_best_figure_dimensions(
    document_width: Union[str, float], scale=1, subplots=(1, 1)
):
    """Set figure dimensions to avoid scaling in LaTeX.
    From: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    document_width: Union[str, float]
            Document textwidth or columnwidth in pts. Predefined strings are also acceptable.
    scale: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = document_width

    # Width of figure (in pts)
    fig_width_pt = width_pt * scale

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


# Plot
COLORS_SPR = ["#137513", "#FF9090", "#13B2FF"]
COLORS_CONV = ["#13B2FF", "#FF9090", "#137513"]
MUSTARD = "#BE9700"

# Reproducibility
SEED = 0

# Data
T = 100

# Spring model
NUM_SPRINGS = 3
SPRING_CONSTANT = np.array([16, 8, 4])
MASS = np.ones(NUM_SPRINGS) * 10
DAMPING_COEF_SPR = np.zeros(NUM_SPRINGS)
DT_SPR = 1  # time step size
INITIAL_STATE_SPR = np.array([[1, 0], [3, 0], [5, 0]])
SD_AA_SPR = 0.1  # noise in the model evolution
SD_O_SPR = 0.1  # noise in the measurement

# Conversation model
NUM_SUBJECTS = 3
SUBJECT_NAMES = ["Bob", "Alice", "Dan"]
FREQ = np.array([1, 0.5, 0.1])
DAMPING_COEF_CONV = np.zeros(NUM_SUBJECTS)
DT_CONV = 0.5  # time step size
INITIAL_STATE_CONV = np.array([[1, 0], [1, 0], [1, 0]])
SD_AA_CONV = 0.1  # noise in the model evolution
SD_O_CONV = 0.01  # noise in the measurement

# Inference
BURN_IN = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 2
NUTS_INIT_METHOD = "jitter+adapt_diag"
TARGET_ACCEPT = 0.9

spring_model = SpringModel(
    num_springs=NUM_SPRINGS,
    spring_constant=SPRING_CONSTANT,
    mass=MASS,
    damping_coefficient=DAMPING_COEF_SPR,
    dt=DT_SPR,
    self_dependent=True,
    sd_mean_uc0=1,
    sd_sd_uc=1,
    mean_mean_a0=np.zeros((NUM_SPRINGS, 2)),
    sd_mean_a0=np.ones((NUM_SPRINGS, 2)) * max(INITIAL_STATE_SPR[:, 0]),
    # Maximum value among initial positions not to make hyperprior too tight
    sd_sd_aa=np.ones(1),
    sd_sd_o=np.ones(1),
    share_sd_aa_across_springs=True,
    share_sd_aa_across_features=True,
    # same variance for position and speed
    share_sd_o_across_springs=True,
    # same measurement noise for different springs
    share_sd_o_across_features=True,
)  # same measurement noise for position and speed

conversation_model = ConversationModel(
    num_subjects=NUM_SUBJECTS,
    frequency=FREQ,
    damping_coefficient=DAMPING_COEF_CONV,
    dt=DT_CONV,
    self_dependent=True,
    sd_mean_uc0=1,
    sd_sd_uc=1,
    mean_mean_a0=np.zeros((NUM_SUBJECTS, 2)),
    sd_mean_a0=np.ones((NUM_SUBJECTS, 2)) * max(INITIAL_STATE_CONV[:, 0]),
    sd_sd_aa=np.ones(1),
    sd_sd_o=np.ones(1),
    share_sd_aa_across_subjects=True,
    share_sd_aa_across_features=True,
    share_sd_o_across_subjects=True,
    share_sd_o_across_features=True,
)


def plot_spring_data(
    ax: Any,
    data: np.ndarray,
    title: str = "",
    line_width: float = 1,
    y_shift_fn: Callable = lambda x, s: x,
):
    num_time_steps = data.shape[-1]

    tt = np.arange(num_time_steps)

    for s in range(NUM_SPRINGS):
        ax.plot(
            tt,
            y_shift_fn(data[s, 0], s),
            label=f"Spring {s + 1}",
            color=COLORS_SPR[s],
            linewidth=line_width,
            linestyle="solid",
        )

    sns.despine(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")


def plot_conversation_data(
    ax: Any,
    data: ConversationSeries,
    title: str = "",
    line_width: float = 1,
    y_shift_fn: Callable = lambda x, s: x,
):
    for s, name in enumerate(SUBJECT_NAMES):
        tt = np.array(
            [t for t, subject in enumerate(data.subjects_in_time) if s == subject]
        )
        ax.plot(
            tt,
            y_shift_fn(data.observation[0, tt], s),
            label=name,
            color=COLORS_CONV[s],
            linewidth=line_width,
            linestyle="solid",
        )
    sns.despine(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")


def conversation_samples_to_evidence(
    samples: ConversationSamples,
) -> ConversationSamples:
    return ConversationSeries(
        subjects_in_time=samples.state.subjects[0],
        prev_time_same_subject=samples.state.prev_time_same_subject[0],
        prev_time_diff_subject=samples.state.prev_time_diff_subject[0],
        observation=samples.observation.values[0],
    )


# A simpler case with serialize was the difference. Someone talks until they are excited and then they
# get embarassed and swich to the different behavior. Self-correcting force like a spring.They are oscillatory.
# They are not monotone. FOr a reason because their vocalic carry emotion and then their self- return to center kind of thing. When they are influence to
# someone. Not out of phase when they started.

# Set parameters for data generation
spring_model.state_space_cpn.parameters.mean_a0.value = INITIAL_STATE_SPR
spring_model.state_space_cpn.parameters.sd_aa.value = np.zeros(1)  # only for plots
spring_model.observation_cpn.parameters.sd_o.value = np.zeros(1)

PLOT_SPRING = True
if PLOT_SPRING:
    # Denoised version is only used for plots
    T_plot = 1000
    coordination = np.zeros((1, T_plot))
    spring_uncoordinated_data = spring_model.draw_samples(
        num_series=1,
        num_time_steps=T_plot,
        coordination_samples=coordination,
        seed=SEED,
    ).observation.values[0]

    coordination = np.ones((1, T_plot)) * 2 / 3
    spring_coordinated_data = spring_model.draw_samples(
        num_series=1,
        num_time_steps=T_plot,
        coordination_samples=coordination,
        seed=SEED,
    ).observation.values[0]

    coordination = np.ones((1, T_plot))
    spring_supercoordinated_data = spring_model.draw_samples(
        num_series=1,
        num_time_steps=T_plot,
        coordination_samples=coordination,
        seed=SEED,
    ).observation.values[0]

    coordination = np.ones((1, T_plot)) * 0.2
    spring_0_2_data = spring_model.draw_samples(
        num_series=1,
        num_time_steps=T_plot,
        coordination_samples=coordination,
        seed=SEED,
    ).observation.values[0]

    coordination = np.ones((1, T_plot)) * 0.8
    spring_0_8_data = spring_model.draw_samples(
        num_series=1,
        num_time_steps=T_plot,
        coordination_samples=coordination,
        seed=SEED,
    ).observation.values[0]

    # Noisy version for inference
    spring_model.state_space_cpn.parameters.sd_aa.value = np.ones(1) * SD_AA_SPR
    spring_model.observation_cpn.parameters.sd_o.value = np.ones(1) * SD_O_SPR

    coordination = np.zeros((1, T))
    noisy_spring_uncoordinated_data = spring_model.draw_samples(
        num_series=1, num_time_steps=T, coordination_samples=coordination, seed=SEED
    ).observation.values[0]

    coordination = np.ones((1, T)) * 2 / 3
    noisy_spring_coordinated_data = spring_model.draw_samples(
        num_series=1, num_time_steps=T, coordination_samples=coordination, seed=SEED
    ).observation.values[0]

    coordination = np.ones((1, T))
    noisy_spring_supercoordinated_data = spring_model.draw_samples(
        num_series=1, num_time_steps=T, coordination_samples=coordination, seed=SEED
    ).observation.values[0]

    coordination = np.ones((1, T)) * 0.2
    noisy_spring_0_2_data = spring_model.draw_samples(
        num_series=1, num_time_steps=T, coordination_samples=coordination, seed=SEED
    ).observation.values[0]

    coordination = np.ones((1, T)) * 0.8
    noisy_spring_0_8_data = spring_model.draw_samples(
        num_series=1, num_time_steps=T, coordination_samples=coordination, seed=SEED
    ).observation.values[0]

    # Plot data
    w, h = calculate_best_figure_dimensions(
        document_width=DOC_WIDTH, scale=1, subplots=(2, 3)
    )
    fig, axs = plt.subplots(2, 3, figsize=(w, h * 1.5))
    axs[0, 1].sharey(axs[0, 0])
    axs[1, 1].sharey(axs[1, 0])

    v_pos = 15
    x_slice = [0, 30]

    plot_spring_data(
        axs[0, 0],
        spring_uncoordinated_data,
        title=f"C = 0",
        y_shift_fn=lambda x, s: x + s * 10,
    )
    axs[0, 0].plot(
        [v_pos, v_pos], [-1, 25], linestyle="dotted", linewidth=0.7, color="black"
    )
    axs[0, 0].set_xlim(x_slice)
    plot_spring_data(
        axs[0, 1],
        spring_0_2_data,
        title=f"C = 0.2",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 10,
    )
    axs[0, 1].plot(
        [v_pos, v_pos], [-1, 25], linestyle="dotted", linewidth=0.7, color="black"
    )
    axs[0, 1].set_xlim(x_slice)
    plot_spring_data(
        axs[0, 2],
        spring_coordinated_data,
        title=f"C = 2/3",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 10,
    )
    axs[0, 2].plot(
        [v_pos, v_pos], [-1, 25], linestyle="dotted", linewidth=0.7, color="black"
    )
    axs[0, 2].set_xlim(x_slice)
    plot_spring_data(
        axs[1, 0],
        spring_0_8_data,
        title=f"C = 0.8",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 10,
    )
    axs[1, 0].plot(
        [v_pos, v_pos], [-1, 25], linestyle="dotted", linewidth=0.7, color="black"
    )
    axs[1, 0].set_xlim(x_slice)
    plot_spring_data(
        axs[1, 1],
        spring_supercoordinated_data,
        title=f"C = 1",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 10,
    )
    axs[1, 1].plot(
        [v_pos, v_pos], [-1, 25], linestyle="dotted", linewidth=0.7, color="black"
    )
    axs[1, 1].set_xlim(x_slice)

    axs[0, 1].set_ylabel("")
    axs[0, 2].set_ylabel("")
    axs[1, 1].set_ylabel("")
    plt.tight_layout()

PLOT_CONVERSATION = False
if PLOT_CONVERSATION:
    # Set parameters for data generation
    conversation_model.state_space_cpn.parameters.mean_a0.value = INITIAL_STATE_CONV
    conversation_model.state_space_cpn.parameters.sd_aa.value = np.zeros(
        1
    )  # only for plots
    conversation_model.observation_cpn.parameters.sd_o.value = np.zeros(1)

    # Denoised version is only used for plots
    T_plot = 1000
    coordination = np.zeros((1, T_plot))
    conversation_uncoordinated_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T_plot,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T_plot)) * 0.5
    conversation_coordinated_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T_plot,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T_plot))
    conversation_supercoordinated_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T_plot,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T_plot)) * 0.2
    conversation_0_2_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T_plot,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T_plot)) * 0.8
    conversation_0_8_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T_plot,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    # Noisy version for inference
    conversation_model.state_space_cpn.parameters.sd_aa.value = np.ones(1) * SD_AA_CONV
    conversation_model.observation_cpn.parameters.sd_o.value = np.ones(1) * SD_O_CONV

    coordination = np.zeros((1, T))
    noisy_conversation_uncoordinated_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T)) * 0.5
    noisy_conversation_coordinated_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T))
    noisy_conversation_supercoordinated_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T)) * 0.2
    noisy_conversation_0_2_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    coordination = np.ones((1, T)) * 0.8
    noisy_conversation_0_8_data = conversation_samples_to_evidence(
        conversation_model.draw_samples(
            num_series=1,
            num_time_steps=T,
            coordination_samples=coordination,
            seed=SEED,
            fixed_subject_sequence=True,
        )
    )

    # Plot data
    w, h = calculate_best_figure_dimensions(
        document_width=DOC_WIDTH, scale=1, subplots=(2, 3)
    )
    fig, axs = plt.subplots(2, 3, figsize=(w, h * 1.5))
    axs[0, 1].sharey(axs[0, 0])
    # axs[1,1].sharey(axs[1,0])

    v_pos = 45
    x_slice = [0, 100]

    plot_conversation_data(
        axs[0, 0],
        conversation_uncoordinated_data,
        title=f"C = 0",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 2.5,
    )
    axs[0, 0].set_xlim(x_slice)
    axs[0, 0].plot(
        [v_pos, v_pos], [-1, 7], linestyle="dotted", linewidth=0.7, color="black"
    )
    plot_conversation_data(
        axs[0, 1],
        conversation_0_2_data,
        title=f"C = 0.2",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 2.5,
    )
    axs[0, 1].set_xlim(x_slice)
    axs[0, 1].plot(
        [v_pos, v_pos], [-1, 7], linestyle="dotted", linewidth=0.7, color="black"
    )
    plot_conversation_data(
        axs[0, 2],
        conversation_coordinated_data,
        title=f"C = 0.5",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 2.5,
    )
    axs[0, 2].set_xlim(x_slice)
    axs[0, 2].plot(
        [v_pos, v_pos], [-1, 7], linestyle="dotted", linewidth=0.7, color="black"
    )
    plot_conversation_data(
        axs[1, 0],
        conversation_0_8_data,
        title=f"C = 0.7",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 2.5,
    )
    axs[1, 0].set_xlim(x_slice)
    axs[1, 0].plot(
        [v_pos, v_pos], [-1, 7], linestyle="dotted", linewidth=0.7, color="black"
    )
    plot_conversation_data(
        axs[1, 1],
        conversation_supercoordinated_data,
        title=f"C = 1",
        line_width=1,
        y_shift_fn=lambda x, s: x + s * 2.5,
    )
    axs[1, 1].set_xlim(x_slice)
    axs[1, 1].plot(
        [v_pos, v_pos], [-1, 7], linestyle="dotted", linewidth=0.7, color="black"
    )

    axs[0, 1].set_ylabel("")
    axs[0, 2].set_ylabel("")
    axs[1, 1].set_ylabel("")
    plt.tight_layout()


def train(
    model: Any,
    evidence: Any,
    init_method: str = NUTS_INIT_METHOD,
    burn_in: int = BURN_IN,
    num_samples: int = NUM_SAMPLES,
    num_chains: int = NUM_CHAINS,
    target_accept: float = TARGET_ACCEPT,
    seed: int = SEED,
):
    # Ignore PyMC warnings
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")

    # The environment variables below will make sure each chain does not take all the resources, slowing down inference.
    os.environ["MKL_NUM_THREADS"] = f"{num_chains}"
    os.environ["OMP_NUM_THREADS"] = f"{num_chains}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{num_chains}"

    model.clear_parameter_values()  # so we can infer them
    _, idata = model.fit(
        evidence=evidence,
        init_method=init_method,
        burn_in=burn_in,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        num_jobs=num_chains,
        target_accept=target_accept,
    )

    posterior_samples = CoordinationPosteriorSamples.from_inference_data(idata)

    # Plot parameter trace
    plot_parameter_trace(model, idata)

    # Plot coordination
    w, h = calculate_best_figure_dimensions(document_width=DOC_WIDTH, scale=1)
    fig = plt.figure(figsize=(w, h))

    posterior_samples.plot(fig.gca(), show_samples=False, line_width=1)
    plt.title("Coordination")
    plt.show()

    return posterior_samples, idata


def plot_parameter_trace(model: Any, idata: Any):
    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()


def build_convergence_summary(idata: Any) -> pd.DataFrame:
    header = ["variable", "mean_rhat", "std_rhat"]

    rhat = az.rhat(idata)
    data = []
    for var, values in rhat.data_vars.items():
        entry = [var, values.to_numpy().mean(), values.to_numpy().std()]
        data.append(entry)

    return pd.DataFrame(data, columns=header)


if __name__ == "__main__":
    # evidence = noisy_spring_uncoordinated_data
    #
    # c_posterior_spring_uncoordinated, idata_spring_uncoordinated = train(spring_model, evidence)
    # build_convergence_summary(idata_spring_uncoordinated)
    plt.show()

    # evidence = noisy_conversation_coordinated_data
    #
    # c_posterior_conversation_coordinated, idata_conversation_coordinated = train(conversation_model, evidence)
    # build_convergence_summary(idata_conversation_coordinated)

    evidence = noisy_spring_coordinated_data

    c_posterior_spring_coordinated, idata_spring_coordinated = train(
        spring_model, evidence
    )
    build_convergence_summary(idata_spring_coordinated)
