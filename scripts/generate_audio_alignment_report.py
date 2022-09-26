import argparse

from coordination.audio.audio import TrialAudio
from coordination.entity.trial import Trial
from coordination.report.audio_alignment_report import AudioAlignmentReport


def generate_report(trial_dir: str, audios_dir: str, out_dir: str):
    trial = Trial.from_directory(trial_dir)
    trial_audio = TrialAudio(trial.metadata, audios_dir)
    report = AudioAlignmentReport(trial, trial_audio, f"Trial: {trial.metadata.number}")
    report.export_to_html(f"{out_dir}/{trial.metadata.number}.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates audio alignment report fof a trial."
    )
    parser.add_argument("--trial_dir", type=str, required=True, help="Directory where serialized trials are located.")
    parser.add_argument("--audios_dir", type=str, required=True, help="Directory where audio files are located.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where the report must be saved.")

    args = parser.parse_args()

    generate_report(args.trial_dir, args.audios_dir, args.out_dir)
