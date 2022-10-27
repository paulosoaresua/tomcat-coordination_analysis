from typing import Optional

import io
import logging
import os

import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter


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


def configure_log(verbose: bool, log_filepath: str):
    if verbose:
        if log_filepath:
            setup_file_logger(log_filepath)
        else:
            setup_custom_logger()
    else:
        logging.disable(logging.CRITICAL)


def image_to_tensorboard(figure: plt.figure):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(figure)
    buffer.seek(0)

    image = PIL.Image.open(buffer)
    image = ToTensor()(image)

    return image


class BaseLogger:

    def __init__(self, measure_suffix: str = ""):
        self.measure_suffix = measure_suffix

    def add_scalar(self, name: str, value: float, step: int):
        # Don't do anything
        pass


class TensorBoardLogger(BaseLogger):

    def __init__(self, out_dir: str, measure_suffix: Optional[str] = None):
        super().__init__(measure_suffix)

        os.makedirs(out_dir, exist_ok=True)

        self.tb_writer = SummaryWriter(out_dir)
        self.measure_suffix = measure_suffix

    def add_scalar(self, name: str, value: float, step: int):
        if self.measure_suffix is not None:
            name = f"{name}_{self.measure_suffix}"
        self.tb_writer.add_scalar(name, value, step)
