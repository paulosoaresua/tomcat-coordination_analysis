import matplotlib.pyplot as plt
import numpy as np

# Custom code
from coordination.component.speech.vocalics_component import VocalicsComponent
from coordination.entity.trial import Trial
from coordination.inference.vocalics import DiscreteCoordinationInferenceFromVocalics, ContinuousCoordinationInferenceFromVocalics
from coordination.plot.coordination import add_discrete_coordination_bar
from coordination.plot.vocalics import plot_vocalic_features


if __name__ == "__main__":
    trial = Trial.from_directory("../data/study-3_2022/T000745/")
    vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics)

    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    vocalics_component.plot_features(axs, 1020, True, True)
