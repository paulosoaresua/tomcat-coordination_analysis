import argparse
from glob import glob
import os
from tqdm import tqdm

from coordination.audio.audio import TrialAudio
from coordination.entity.trial import Trial
from coordination.report.audio_alignment_report import AudioAlignmentReport


def generate_single_report(trial_dir: str, audios_dir: str, out_dir: str):
    trial = Trial.from_directory(trial_dir)
    trial_audio = TrialAudio(trial.metadata, audios_dir)
    report = AudioAlignmentReport(trial, trial_audio, f"Trial: {trial.metadata.number}")
    report.export_to_html(f"{out_dir}/{trial.metadata.number}/audio_alignment_report.html")


def generate_reports(input_dir: str, audios_dir: str, out_dir: str, multi: bool):
    if not os.path.exists(input_dir):
        raise Exception(f"Directory {input_dir} does not exist.")

    if multi:
        filepaths = glob(f"{input_dir}/T*")
        for i, filepath in tqdm(enumerate(filepaths), desc="Generating reports..."):
            if os.path.isdir(filepath):
                generate_single_report(filepath, audios_dir, out_dir)
    else:
        generate_single_report(input_dir, audios_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates audio alignment report fof a trial."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory where a single serialized trial is located or directory where " 
                             "multiple trial directories are located in case the multi option is enabled.")
    parser.add_argument("--audios_dir", type=str, required=True, help="Directory where audio files are located.")
    parser.add_argument("--out_dir", type=str, required=True, help="File or directory where the report must be saved.")
    parser.add_argument("--multi", action="store_true", required=False, default=False,
                        help="Whether input_dir contains multiple trial directories to be processed.")

    args = parser.parse_args()

    generate_reports(args.input_dir, args.audios_dir, args.out_dir, args.multi)
