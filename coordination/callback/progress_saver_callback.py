import json

from pymc.backends.ndarray import NDArray
from pymc.sampling.parallel import Draw


class ProgressSaverCallback:
    """
    This class represents a callback to be used during the model fit function to save the inference
    progress to a directory.
    """

    def __init__(self, out_dir: str, saving_frequency: int):
        """
        Creates a progress saver callback.

        @param out_dir: directory where progress must be saved.
        @param saving_frequency: a number defining the number of samples between each progress
            saving. For instance, if saving_frequency = 100, it will save progress at every 100
            samples drawn.
        """
        self.out_dir = out_dir
        self.saving_frequency = saving_frequency
        self._progress_filepath = f"{self.out_dir}/progress.json"
        self.progress_dict = {"step": {}, "log_prob": {}, "num_divergences": {}}

    def __call__(self, trace: NDArray, draw: Draw):
        """
        Saves the progress to a file. This function called at every sample during a model fit.

        @param trace: inference trace.
        @param draw: information about the draw.
        """
        if len(trace) % self.saving_frequency == 0:
            key = f"chain{draw.chain}"
            self.progress_dict["step"][key] = len(trace)
            self.progress_dict["log_prob"][key] = float(draw.stats[0]["model_logp"])
            if key in self.progress_dict["num_divergences"]:
                self.progress_dict["num_divergences"][key] += int(
                    draw.stats[0]["diverging"]
                )
            else:
                self.progress_dict["num_divergences"][key] = int(
                    draw.stats[0]["diverging"]
                )

            with open(self._progress_filepath, "w") as f:
                # We open the file every time something is written to it instead of keeping it
                # open so we can erase the previous information recorded.
                json.dump(self.progress_dict, f)
