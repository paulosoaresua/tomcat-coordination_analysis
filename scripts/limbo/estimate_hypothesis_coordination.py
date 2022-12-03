import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

root_dir = f"/Users/paulosoares/code/tomcat-coordination/data/study-3_2022/tomcat_agent/estimates/gaussian_latent"
with open(f"{root_dir}/mission1/coordination/estimation/estimates.pkl", "rb") as f:
    tomcat_params_mission1 = pickle.load(f)

with open(f"{root_dir}/mission2/coordination/estimation/estimates.pkl", "rb") as f:
    tomcat_params_mission2 = pickle.load(f)

tomcat_coordination_mission1 = np.array([np.mean(means) for means, _ in tomcat_params_mission1])
tomcat_coordination_mission2 = np.array([np.mean(means) for means, _ in tomcat_params_mission2])
tomcat_coordination_all_missions = np.concatenate([tomcat_coordination_mission1, tomcat_coordination_mission2])

root_dir = f"/Users/paulosoares/code/tomcat-coordination/data/study-3_2022/no_advisor/estimates/gaussian_latent"
with open(f"{root_dir}/mission1/coordination/estimation/estimates.pkl", "rb") as f:
    no_advisor_params_mission1 = pickle.load(f)

with open(f"{root_dir}/mission2/coordination/estimation/estimates.pkl", "rb") as f:
    no_advisor_params_mission2 = pickle.load(f)

no_advisor_coordination_mission1 = np.array([np.mean(means) for means, _ in no_advisor_params_mission1])
no_advisor_coordination_mission2 = np.array([np.mean(means) for means, _ in no_advisor_params_mission2])
no_advisor_coordination_all_missions = np.concatenate(
    [no_advisor_coordination_mission1, no_advisor_coordination_mission2])

print("Mission 1")
print(np.var(tomcat_coordination_mission1), np.var(no_advisor_coordination_mission1))
print(ttest_ind(tomcat_coordination_mission1, no_advisor_coordination_mission1, alternative="greater"))

bp_mission1 = [no_advisor_coordination_mission1, tomcat_coordination_mission1]
plt.figure(figsize=(6, 6))
bp = plt.boxplot(bp_mission1, patch_artist=True, labels=["No Advisor", "ToMCAT"])
plt.title("Mission 1", fontsize=16, fontweight="bold")
plt.ylabel("Average Coordination")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("/Users/paulosoares/Desktop/mission1.png")


bp_mission2 = [no_advisor_coordination_mission2, tomcat_coordination_mission2]
plt.figure(figsize=(6, 6))
plt.boxplot(bp_mission2, patch_artist=True, labels=["No Advisor", "ToMCAT"])
plt.title("Mission 2", fontsize=16, fontweight="bold")
plt.ylabel("Average Coordination")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("/Users/paulosoares/Desktop/mission2.png")

print("\nMission 2")
print(np.var(tomcat_coordination_mission2), np.var(no_advisor_coordination_mission2))
print(ttest_ind(tomcat_coordination_mission2, no_advisor_coordination_mission2, alternative="greater"))

bp_all_missions = [no_advisor_coordination_all_missions, tomcat_coordination_all_missions]
plt.figure(figsize=(6, 6))
plt.boxplot(bp_all_missions, patch_artist=True, labels=["No Advisor", "ToMCAT"])
plt.title("Mission 1 + Mission 2", fontsize=16, fontweight="bold")
plt.ylabel("Average Coordination")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig("/Users/paulosoares/Desktop/all_missions.png")

print("\nAll Missions")
print(np.var(tomcat_coordination_all_missions), np.var(no_advisor_coordination_all_missions))
print(ttest_ind(tomcat_coordination_all_missions, no_advisor_coordination_all_missions, alternative="greater"))

#
# def estimate_regression(coordination_path: str, dataset_path: str, plot_out_path: str):
#     input_features: InputFeaturesDataset
#     scores: np.ndarray
#     with open(dataset_path, "rb") as f:
#         input_features, scores = pickle.load(f)
#
#     with open(coordination_path, "rb") as f:
#         params = pickle.load(f)
#
#     coordination = np.array([np.mean(means) for means, variances in params])
#
#     r, p = pearsonr(coordination, scores)
#     regressor = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True, alpha_init=1, lambda_init=1e-3)
#     regressor.fit(coordination[:, np.newaxis], scores)
#
#     label = f"$r = {r:.2f}, p = {p:.4f}$"
#     fig = plot_coordination_vs_score_regression(coordination, scores, regressor, label)
#     plt.savefig(plot_out_path)
#     plt.tight_layout()
#     plt.close(fig)
#
#
