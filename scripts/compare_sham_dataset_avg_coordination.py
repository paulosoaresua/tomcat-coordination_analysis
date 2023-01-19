from typing import List

import argparse
import os
import pickle

import numpy as np

from coordination.common.utils import set_random_seed
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
from coordination.model.gendered_beta_coordination_blending_latent_vocalics import \
    GenderedBetaCoordinationBlendingLatentVocalics
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset


def compare(model_path: str, dataset_path: str, num_particles: int, seed: int, num_jobs: int, out_dir: str,
          features: List[str], gendered: bool, link: bool, num_shuffles: int):

    # Loading dataset
    with open(dataset_path, "rb") as f:
        dataset = BetaCoordinationLatentVocalicsDataset.from_latent_vocalics_dataset(pickle.load(f))

    dataset.keep_vocalic_features(features)

    if not link:
        dataset.disable_speech_semantic_links()

    if gendered:
        dataset.normalize_gender()
        model = GenderedBetaCoordinationBlendingLatentVocalics.from_pickled_file(model_path)
    else:
        dataset.normalize_per_subject()
        model = BetaCoordinationBlendingLatentVocalics.from_pickled_file(model_path)

    avg_coordinations = np.zeros((0, 2))

    print("Real Dataset")
    real_summaries = model.predict(evidence=dataset, num_particles=num_particles, seed=seed, num_jobs=num_jobs)
    real_avg_coordination = [np.mean(s.coordination_mean) for s in real_summaries]

    print("")
    print("Sham Dataset")
    set_random_seed(seed)
    for i in range(num_shuffles):
        print("")
        print(f"~~~~~ SHUFFLE {i + 1}/{num_shuffles} ~~~~~")
        dataset.shuffle()
        sham_summaries = model.predict(evidence=dataset, num_particles=num_particles, seed=seed, num_jobs=num_jobs)
        sham_avg_coordination = [np.mean(s.coordination_mean) for s in sham_summaries]

        new_avg_pairs = np.array([real_avg_coordination, sham_avg_coordination]).T
        avg_coordinations = np.vstack([avg_coordinations, new_avg_pairs])

    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(f"{out_dir}/real_vs_sham.txt", avg_coordinations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infers coordination given a trained model and a dataset of observed vocalics over time."
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to a pre-trained coordination model.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset used to train the coordination model.")
    parser.add_argument("--n_particles", type=int, required=False, default=10000,
                        help="Number of particles used for inference.")
    parser.add_argument("--seed", type=int, required=False, default=0, help="Random seed for replication.")
    parser.add_argument("--n_jobs", type=int, required=False, default=1,
                        help="Number of jobs to infer trials in parallel.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where to save the inferences.")
    parser.add_argument("--features", type=str, required=False, default="pitch, intensity, jitter, shimmer",
                        help="List of vocalic features to consider. It can be any subset of the default value.",)
    parser.add_argument("--gendered", type=int, required=False, default=0,
                        help="Whether to use a model that considers speakers' genders.")
    parser.add_argument("--link", type=int, required=False, default=0,
                        help="Whether to use a model that considers speech semantic link.")
    parser.add_argument("--num_shuffles", type=int, required=False, default=10,
                        help="Number of shuffles to compare with real data.")

    args = parser.parse_args()

    def format_feature_name(name: str):
        return name.strip().lower()

    compare(model_path=args.model_path,
          dataset_path=args.dataset_path,
          num_particles=args.n_particles,
          seed=args.seed,
          num_jobs=args.n_jobs,
          out_dir=args.out_dir,
          features=list(map(format_feature_name, args.features.split(","))),
          gendered=args.gendered > 0,
          link=args.link > 0,
          num_shuffles=args.num_shuffles)
