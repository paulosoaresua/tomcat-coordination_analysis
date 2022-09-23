import logging
import os


def _setup_logger(handler):
    logger = logging.getLogger()
    for old_handler in logger.handlers:
        logger.removeHandler(old_handler)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)  # Prints all types of messages including DEBUG ones
    logger.addHandler(handler)


def setup_custom_logger():
    _setup_logger(logging.StreamHandler())


def setup_file_logger(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    _setup_logger(logging.FileHandler(filepath))
