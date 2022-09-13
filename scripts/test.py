from coordination.audio.audio import TrialAudio, VocalicsComponentAudio
from coordination.entity.trial import Trial
from coordination.component.speech.vocalics_component import VocalicsComponent
from coordination.report.coordination_abrupt_change_report import CoordinationAbruptChangeReport



NUM_TIME_STEPS = 100


if __name__ == "__main__":
    trial = Trial.from_directory("../data/study-3_2022/T000745/")
    vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics)
    vocalics_a, vocalics_b = vocalics_component.sparse_series(NUM_TIME_STEPS)
    vocalics_a.normalize()
    vocalics_b.normalize()

    trial_audio = TrialAudio(trial.metadata, "/Users/paulosoares/data/study-3_2022/audio")
    audio_component = VocalicsComponentAudio.from_vocalics_component(trial_audio, vocalics_component)
    audio_a, audio_b = audio_component.sparse_series(NUM_TIME_STEPS)
    # report = CoordinationAbruptChangeReport(mean_cs, vocalics_a, vocalics_b, audio_a, audio_b)