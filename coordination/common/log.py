import logging
import os
from typing import Optional


def _setup_logger(handler):
    """
    Configures logger with a proper format.

    @param handler: an output stream that handles where the logs are saved.
    """

    logger = logging.getLogger()
    for old_handler in logger.handlers:
        logger.removeHandler(old_handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)  # Prints all types of messages including DEBUG ones
    logger.addHandler(handler)


def setup_custom_logger():
    """
    Configures a logger that writes to the terminal.
    """
    _setup_logger(logging.StreamHandler())


def setup_file_logger(filepath: str):
    """
    Configures a logger that writes to a file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    _setup_logger(logging.FileHandler(filepath))


def configure_log(verbose: bool, log_filepath: Optional[str] = None, level: int = 0):
    """
    Configures a logger depending on verbosity.

    @param verbose: if True, all messages are logged; otherwise, just the critical ones are logged.
    @param log_filepath: path to the file where logs must be saved. If not provided, logs will be
    written to the terminal.
    @param level: 0 means log everything, 1 disable debug messages, 2 disable warning messages.
    """
    if verbose:
        if log_filepath:
            setup_file_logger(log_filepath)
        else:
            setup_custom_logger()

        if level > 0:
            logging.disable(logging.DEBUG)
        if level > 1:
            logging.disable(logging.WARNING)
            logging.disable(logging.WARN)
    else:
        logging.disable(logging.CRITICAL)
