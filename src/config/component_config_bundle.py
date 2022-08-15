import json
import os


class DataBaseConfig:
    def __init__(self, address: str, port: int, database: str):
        self.address = address
        self.port = port
        self.database = database


class VocalicsConfig:
    """
    Any configuration related to the vocalics component
    """
    no_vocalics = False  # always read vocalics if available
    feature_map = {"pitch": "f0final_sma", "intensity": "wave_rmsenergy_sma"}
    database_config = DataBaseConfig("localhost", 5432, "asist_vocalics")


class ComponentConfigBundle:
    """
    Configurations from all the components used in the coordination model
    """
    vocalics_config = VocalicsConfig()

    def save(self, out_dir: str):
        config_dict = {
            "vocalics": self.vocalics_config.__dict__
        }

        with open(f"{out_dir}/config_bundle.json", "w") as f:
            json.dump(config_dict, f, indent=4)

    def load(self, config_dir: str):
        config_path = f"{config_dir}/config_bundle.json"
        if not os.path.exists(config_path):
            raise Exception(f"Could not find the file config_bundle.json in {config_dir}.")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

            for k, v in config_dict["vocalics"].items():
                setattr(self.vocalics_config, k, v)
