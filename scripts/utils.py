import argparse
import logging
import os

import coordination.common.log as log
from coordination.config.database_config import DatabaseConfig
from coordination.entity.trial import Trial
from coordination.loader.vocalics_reader import VocalicFeature


def get_common_parse_and_save_metadata_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where post-parsed data must be saved.")
    parser.add_argument("--verbose", action="store_true", required=False, default=False,
                        help="Whether to print the logs.")
    parser.add_argument("--log_dir", type=str, required=False,
                        help="""Directory where log files must be saved. If not provided, logs will be printed to the 
                           terminal if the --verbose option is active.""")
    parser.add_argument("--vocalics_database_address", type=str, required=False, default="localhost",
                        help="Address of the database server containing the vocalic features.")
    parser.add_argument("--vocalics_database_port", type=int, required=False, default=5432,
                        help="Port of the database server containing the vocalic features.")
    parser.add_argument("--vocalics_database_name", type=str, required=False, default="asist_vocalics",
                        help="Name of the database containing the vocalic features.")

    return parser


def configure_log(verbose: bool, log_filepath: str):
    if verbose:
        if log_filepath:
            log.setup_file_logger(log_filepath)
        else:
            log.setup_custom_logger()
    else:
        logging.disable(logging.CRITICAL)


def parse_and_save_metadata(metadata_filepath: str, trial_out_dir: str, verbose: bool, log_dir: str,
                            database_config: DatabaseConfig):
    log_filepath = ""
    if log_dir != "":
        filename = os.path.basename(metadata_filepath).rsplit('.')[0]
        log_filepath = f"{log_dir}/{filename}.txt"

    configure_log(verbose, log_filepath)
    trial = Trial.from_metadata_file(metadata_filepath, database_config,
                                     [VocalicFeature.PITCH, VocalicFeature.INTENSITY])
    trial.save(trial_out_dir)
