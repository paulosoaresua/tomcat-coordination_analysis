import argparse
import numpy as np
import os
import pickle

from coordination.common.dataset import EvidenceDataset


def merge_dataset(dataset1_path: str, dataset2_path: str, out_path: str):
    with open(dataset1_path, "rb") as f:
        dataset1 = pickle.load(f)

    with open(dataset2_path, "rb") as f:
        dataset2 = pickle.load(f)

    dataset = dataset1.merge(dataset2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset of vocalics for mission 1, mission 2 and both missions for all the serialized "
                    "trials in a given folder."
    )

    parser.add_argument("--dataset1_path", type=str, required=True,
                        help="Path containing the first dataset.")
    parser.add_argument("--dataset2_path", type=str, required=True,
                        help="Path containing the second dataset.")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path where the merged dataset must be saved.")

    args = parser.parse_args()
    merge_dataset(args.dataset1_path, args.dataset2_path, args.out_path)
