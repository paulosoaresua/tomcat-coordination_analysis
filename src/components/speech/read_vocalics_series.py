import json
import os
import pickle
from glob import glob
from typing import Any, Dict


def read_vocalics_series(vocalics_series_dir: str) -> Dict[str, Any]:
    vocalics_series_data = {}

    for vocalics_series_path in glob(vocalics_series_dir + "/*"):
        trial_number = os.path.basename(vocalics_series_path)

        vocalics_series_data[trial_number] = {
            "trial_info": None,
            "series_a": None,
            "series_b": None
        }

        with open(f"{vocalics_series_path}/trial_info.json", "rb") as f:
            vocalics_series_data[trial_number]["trial_info"] = json.load(f)

        vocalics_path = vocalics_series_path + "/vocalics"
        with open(f"{vocalics_path}/series_a.pkl", "rb") as f:
            vocalics_series_data[trial_number]["series_a"] = pickle.load(f)
        with open(f"{vocalics_path}/series_b.pkl", "rb") as f:
            vocalics_series_data[trial_number]["series_b"] = pickle.load(f)

    return vocalics_series_data
