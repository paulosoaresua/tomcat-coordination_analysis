import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Custom code
from coordination.audio.audio import TrialAudio
from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent
from coordination.entity.trial import Trial
from coordination.inference.vocalics import DiscreteCoordinationInferenceFromVocalics, ContinuousCoordinationInferenceFromVocalics
from coordination.plot.coordination import add_discrete_coordination_bar
from coordination.plot.vocalics import plot_vocalic_features
from coordination.report.coordination_change_report import CoordinationChangeReport


if __name__ == "__main__":
    NUM_TIME_STEPS = 17 * 60
    trial = Trial.from_directory("../data/study-3_2022/T000745/")
    vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics,
                                                         segmentation_method=SegmentationMethod.KEEP_ALL)

    vocalic_series = vocalics_component.sparse_series(NUM_TIME_STEPS, trial.metadata.mission_start)
    vocalic_series.normalize_per_subject()
