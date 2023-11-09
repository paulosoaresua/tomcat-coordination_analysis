from typing import Any, List, Optional

import numpy as np

from coordination.module.serial_component import SerialComponent, SerialComponentSamples
from coordination.module.serial_observation import SerialObservation, SerialObservationSamples


















class ComponentSamples:

    def __init__(self,
                 latent_samples: SerialComponentSamples,
                 data_samples: SerialObservationSamples):
        """
        Creates a collection of samples for the component.

        @param latent_samples: Samples generated by the latent variables.
        @param data_samples: Samples generated by the observed variables.
        """
        self.latent_samples = latent_samples
        self.data_samples = data_samples


class Evidence:
    """
    This class encapsulates observations and metadata.
    """

    def __init__(self,
                 uuid: str,
                 features: List[str],
                 num_time_steps_in_coordination_scale: int,
                 subjects_in_time: np.ndarray,
                 observation: np.ndarray,
                 previous_time_same_subject: np.ndarray,
                 previous_time_diff_subject: np.ndarray,
                 time_steps_in_coordination_scale: np.ndarray):
        """
        Creates evidence to the component.

        @param uuid: Unique id identifying the data.
        @param features: Feature names, this should match the data_dimension_names defined in the
        component.
        @param num_time_steps_in_coordination_scale: The dimension size of the coordination
        variable.
        @param subjects_in_time: An array of subject indices in the scale of coordination. If
        there's no observation in a specific time, the subject at that time step must be -1.
        @param observation:
        @param previous_time_same_subject:
        @param previous_time_diff_subject:
        @param time_steps_in_coordination_scale:
        """
        self.uuid = uuid
        self.features = features
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.subjects_in_time = subjects_in_time
        self.observation = observation
        self.previous_time_same_subject = previous_time_same_subject
        self.previous_time_diff_subject = previous_time_diff_subject
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

    def normalize_per_subject(self):
        """
        Scale measurements to have mean 0 and standard deviation 1 per subject and feature.
        """
        all_subjects = set(self.subjects_in_time)

        for subject in all_subjects:
            obs_per_subject = self.observation[:, self.subjects_in_time == subject]
            mean = obs_per_subject.mean(axis=1)[:, None]
            std = obs_per_subject.std(axis=1)[:, None]
            self.observation[:, self.subjects_in_time == subject] = (obs_per_subject - mean) / std

    def plot_observations(self, axs: List[Any]):
        all_subjects = set(self.subjects_in_time)

        for vocalic_feature_idx in range(min(self.num_vocalic_features, len(axs))):
            ax = axs[vocalic_feature_idx]
            for subject in all_subjects:
                subject_mask = self.subjects_in_time == subject
                xs = self.time_steps_in_coordination_scale[subject_mask]
                ys = self.observation[vocalic_feature_idx, subject_mask]
                if len(xs) == 1:
                    ax.scatter(xs, ys, label=subject)
                else:
                    ax.plot(xs, ys, label=subject, marker="o")

            ax.set_title(self.features[vocalic_feature_idx])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Observed Value")
            ax.set_xlim([-0.5, self.num_time_steps_in_coordination_scale + 0.5])
            ax.legend()

    @classmethod
    def from_data_frame(cls, evidence_df: pd.DataFrame, vocalic_features: List[str]):
        """
        Parses a dataframe entry to create an evidence object that can be used to fit the model.
        """

        obs_vocalic = []
        for vocalic_feature in vocalic_features:
            obs_vocalic.append(np.array(literal_eval(evidence_df[f"{vocalic_feature}"].values[0])))
        obs_vocalic = np.array(obs_vocalic)

        return cls(
            uuid=evidence_df["experiment_id"].values[0],
            features=vocalic_features,
            num_time_steps_in_coordination_scale=evidence_df["num_time_steps_in_coordination_scale"].values[0],
            subjects_in_time=np.array(literal_eval(evidence_df["vocalic_subjects"].values[0]), dtype=int),
            observation=obs_vocalic,
            previous_time_same_subject=np.array(
                literal_eval(evidence_df["vocalic_previous_time_same_subject"].values[0]), dtype=int),
            previous_time_diff_subject=np.array(
                literal_eval(evidence_df["vocalic_previous_time_diff_subject"].values[0]), dtype=int),
            time_steps_in_coordination_scale=np.array(
                literal_eval(evidence_df["vocalic_time_steps_in_coordination_scale"].values[0]), dtype=int)
        )

    @property
    def num_time_steps_in_vocalic_scale(self) -> int:
        return self.observation.shape[-1]

    @property
    def num_vocalic_features(self) -> int:
        return self.observation.shape[-2]

    @property
    def vocalic_prev_same_subject_mask(self) -> np.ndarray:
        return np.where(self.previous_time_same_subject >= 0, 1, 0)

    @property
    def vocalic_prev_diff_subject_mask(self) -> np.ndarray:
        return np.where(self.previous_time_diff_subject >= 0, 1, 0)


class GenericComponent:

    def __init__(self,
                 latent_name: str,
                 observation_name: str,
                 num_subjects: int,
                 latent_dimension_size: int,
                 data_dimension_size: int,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 sd_sd_o: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_dimensions: bool,
                 share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_dimensions: bool,
                 share_sd_o_across_subjects: bool,
                 share_sd_o_across_dimensions: bool,
                 latent_dimension_names: Optional[List[str]] = None,
                 data_dimension_names: Optional[List[str]] = None):
        """
        Creates a component for the coordination model. A component is composed of its latent
        and observable representation, and is defined for a group of subjects.

        @param latent_name: Name of the latent variables.
        @param observation_name: Name of the observed variables.
        @param num_subjects: The number of subjects that possess the component.
        @param latent_dimension_size: The number of dimensions in the latent variables.
        @param data_dimension_size: The number of dimensions in the observed variables.
        @param self_dependent: Whether the latent variables are tied to the previous value from
        the same subject. If False, coordination will blend the previous latent value of a different
        subject with the initial value of the latent variable for the current subject (the latent
        variable's prior for that subject).
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
        latent variables).
        @param sd_mean_a0: std of the hyper-prior of mu_a0.
        @param sd_sd_aa: std of the hyper-prior of sigma_aa (std of the Gaussian random walk of
        the latent variables.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the ).
        @param share_mean_a0_across_subjects: Whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: Whether to use the same mu_a0 for all dimensions.
        @param share_sd_aa_across_subjects: Whether to use the same sigma_aa for all subjects.
        @param share_sd_aa_across_dimensions: Whether to use the same sigma_aa for all dimensions.
        @param share_sd_o_across_subjects: Whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: Whether to use the same sigma_o for all dimensions.
        @param latent_dimension_names: The names of each dimension of the latent variables. If not
        informed, it will be filled with numbers 0,1,2 up to the dimensionality of the latent
        variables.
        @param data_dimension_names: The names of each dimension of the observed component. If not
        informed, it will be filled with numbers 0,1,2 up to the dimensionality of the observed
        variables.
        """

        self.num_subjects = num_subjects
        self.latent_dimension_names = latent_dimension_names
        self.data_dimension_names = data_dimension_names

        self.latent_variables = SerialComponent(uuid=latent_name,
                                                num_subjects=num_subjects,
                                                dim_value=latent_dimension_size,
                                                self_dependent=self_dependent,
                                                mean_mean_a0=mean_mean_a0,
                                                sd_mean_a0=sd_mean_a0,
                                                sd_sd_aa=sd_sd_aa,
                                                share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                                                share_mean_a0_across_features=share_mean_a0_across_dimensions,
                                                share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                                                share_sd_aa_across_features=share_sd_aa_across_dimensions)

        self.observed_variables = SerialObservation(uuid=observation_name,
                                                    num_subjects=num_subjects,
                                                    dim_value=data_dimension_size,
                                                    sd_sd_o=sd_sd_o,
                                                    share_sd_o_across_subjects=share_sd_o_across_subjects,
                                                    share_sd_o_across_features=share_sd_o_across_dimensions)

    @property
    def parameter_names(self) -> List[str]:
        """
        Get the list of latent parameters. Some parameters may not be latent if they are fixed in
        the model. Those won't have their names in the list.

        @return: Latent parameter names.
        """
        names = self.latent_variables.parameter_names
        names.extend(self.observed_variables.parameter_names)

        return names

    def draw_samples(self,
                     coordination_samples: np.array,
                     time_scale_density: float,
                     can_repeat_subject: bool,
                     fixed_subject_sequence: bool,
                     seed: Optional[int] = None) -> GenericComponentSamples:
        """
        Sample values for the latent and observed variables using ancestral sampling.

        @param coordination_samples: Coordination values to use for the blending strategy when
        generating samples for the component.
        @param time_scale_density: How often do we observe the component. The value should range
        between 0 and 1, where 1 means we generate samples at every time step in the coordination
        time scale.
        @param can_repeat_subject: Whether we can repeat subjects between subsequent samples.
        @param fixed_subject_sequence: Whether the sequence of subjects is fixed (one after another
        circling back to the first), or random.
        @param seed: Random seed for reproducibility.

        @return: Samples for the latent and observed variables in the component.
        """
        num_series, num_time_steps = coordination_samples.shape
        latent_samples = self.latent_variables.draw_samples(num_series=num_series,
                                                            time_scale_density=time_scale_density,
                                                            coordination=coordination_samples,
                                                            can_repeat_subject=can_repeat_subject,
                                                            fixed_subject_sequence=fixed_subject_sequence,
                                                            seed=seed)

        data_samples = self.observed_variables.draw_samples(
            latent_component=latent_samples.values,
            subjects=latent_samples.subjects,
            seed=seed)

        samples = GenericComponentSamples(latent_samples=latent_samples,
                                          data_samples=data_samples)

        return samples

    def update_pymc_model(self,
                          coordination: Any,
                          evidence: VocalicSeries):
        latent = self.latent_variables.update_pymc_model(
            coordination=coordination[evidence.time_steps_in_coordination_scale],
            prev_time_same_subject=evidence.previous_time_same_subject,
            prev_time_diff_subject=evidence.previous_time_diff_subject,
            prev_same_subject_mask=evidence.vocalic_prev_same_subject_mask,
            prev_diff_subject_mask=evidence.vocalic_prev_diff_subject_mask,
            subjects=evidence.subjects_in_time,
            time_dimension=f"{self.latent_variables.uuid}_time",
            feature_dimension=f"{self.latent_variables.uuid}_dimension")[0]

        self.observed_variables.update_pymc_model(latent_component=latent,
                                                  feature_dimension=f"{self.observed_variables.uuid}_dimension",
                                                  time_dimension=f"{self.observed_variables.uuid}_time",
                                                  subjects=evidence.subjects_in_time,
                                                  observed_values=evidence.observation)