from glob import glob
import logging
import os

from coordination.config.database_config import DatabaseConfig
from scripts.utils import get_common_parse_and_save_metadata_argument_parser, parse_and_save_metadata


def parse_and_save_metadata_from_dir(metadata_dir: str, trial_out_dir: str, verbose: bool, log_dir: str,
                                     database_config: DatabaseConfig):

    if not os.path.exists(metadata_dir):
        raise Exception(f"Directory {metadata_dir} does not exist.")

    filepaths = glob(f"{metadata_dir}/*.metadata")
    for i, filepath in enumerate(filepaths):
        # I print the progress to the screen because progress bars are used internally for each metadata file and nested
        # progress bars do not work well.
        if i > 0:
            print("")

        print(f"[{i + 1}/{len(filepaths)}]: {os.path.basename(filepath)}")
        parse_and_save_metadata(filepath, trial_out_dir, verbose, log_dir, database_config)


if __name__ == "__main__":
    description = """
            Parses a collection of Minecraft .metadata files in a directory to extract relevant data to the coordination 
            model and saves the post processed trial structures to different folders; one for each parsed .metadata 
            file."""

    parser = get_common_parse_and_save_metadata_argument_parser(description)
    parser.add_argument("--metadata_dir", type=str, required=False,
                        help="Directory containing a collection of .metadata files to be parsed.")

    args = parser.parse_args()
    database_config = DatabaseConfig(args.vocalics_database_address, args.vocalics_database_port,
                                     args.vocalics_database_name)

    parse_and_save_metadata_from_dir(args.metadata_dir, args.out_dir, args.verbose, args.log_dir, database_config)
