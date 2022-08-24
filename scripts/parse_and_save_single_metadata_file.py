from coordination.config.database_config import DatabaseConfig
from scripts.utils import get_common_parse_and_save_metadata_argument_parser, parse_and_save_metadata

if __name__ == "__main__":
    description = """
        Parses a Minecraft .metadata file to extract relevant data to the coordination model and saves the post
        processed trial structures to a folder."""

    parser = get_common_parse_and_save_metadata_argument_parser(description)
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the .metadata file to be parsed.")

    args = parser.parse_args()
    database_config = DatabaseConfig(args.vocalics_database_address, args.vocalics_database_port,
                                     args.vocalics_database_name)
    parse_and_save_metadata(args.metadata_path, args.out_dir, args.verbose, args.log_dir, database_config)
