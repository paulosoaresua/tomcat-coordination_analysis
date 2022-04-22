from src.trial import Trial
from src.components.speech.vocalics_aggregator import VocalicsAggregator
from src.components.speech.vocalics_writer import VocalicsWriter

if __name__ == "__main__":
    trial = Trial()
    trial.parse("../data/asist/study2/metadata/.metadata")

    aggregator = VocalicsAggregator(trial.utterances_per_subject)
    component = aggregator.split_with_current_utterance_truncation()
    VocalicsWriter().write("../data/asist/study2/evidence", component, trial.start, 1/1000)
