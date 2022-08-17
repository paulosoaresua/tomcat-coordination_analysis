import logging


def _setup_logger(handler):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)


def setup_custom_logger():
    _setup_logger(logging.StreamHandler())


def setup_file_logger(filepath: str):
    _setup_logger(logging.FileHandler(filepath))
