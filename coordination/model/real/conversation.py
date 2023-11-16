from coordination.model.model import Model
from coordination.module.coordination.sigmoid_gaussian_coordination import \
    SigmoidGaussianCoordination
from coordination.module.latent_component.serial_mass_spring_damper_latent_component import \
    SerialMassSpringDamperLatentComponent
from coordination.module.observation.serial_gaussian_observation import SerialGaussianObservation
from coordination.model.synthetic.constants import (
    NUM_SUBJECTS,
    SUBJECT_NAMES_CONVERSATION_MODEL,
    ANGULAR_FREQUENCIES_CONVERSATION_MODEL,
    DAMPENING_COEFFICIENTS_CONVERSATION_MODEL,
    DT_CONVERSATION_MODEL,
    INITIAL_STATE_CONVERSATION_MODEL,
    SD_MEAN_UC0,
    SD_SD_UC,
    MEAN_MEAN_A0,
    SD_MEAN_A0_CONVERSATION_MODEL,
    SD_SD_A,
    SD_SD_O,
    SHARE_MEAN_A0_ACROSS_SUBJECT,
    SHARE_MEAN_A0_ACROSS_DIMENSIONS,
    SHARE_SD_ACROSS_SUBJECTS,
    SHARE_SD_ACROSS_DIMENSIONS
)


class ConversationModel(Model):

    def __init__(
            self,
            name: str,
            pymc_model: pm.Model,
            num_subjects: int = NUM_SUBJECTS_CONVERSATION_MODEL,
            squared_angular_frequency: np.ndarray = ANGULAR_FREQUENCIES_CONVERSATION_MODEL,
            dampening_coefficient: np.ndarray = DAMPENING_COEFFICIENTS_CONVERSATION_MODEL,
            dt: float = DT_CONVERSATION_MODEL,
            sd_mean_uc0: float = SD_MEAN_UC0_CONVERSATION_MODEL,
            sd_sd_uc: float = SD_SD_UC_CONVERSATION_MODEL,
            mean_mean_a0: np.ndarray = MEAN_MEAN_A0_CONVERSATION_MODEL,
            sd_mean_a0: np.ndarray = SD_MEAN_A0_CONVERSATION_MODEL,
            sd_sd_a: np.ndarray = SD_SD_A_CONVERSATION_MODEL,
            sd_sd_o: np.ndarray = SD_SD_O_CONVERSATION_MODEL,
            share_mean_a0_across_subjects: bool = SHARE_MEAN_A0_ACROSS_SUBJECT_CONVERSATION_MODEL,
            share_mean_a0_across_dimensions: bool = SHARE_MEAN_A0_ACROSS_DIMENSIONS_CONVERSATION_MODEL,
            share_sd_a_across_subjects: bool = SHARE_SD_ACROSS_SUBJECTS_CONVERSATION_MODEL,
            share_sd_a_across_dimensions: bool = SHARE_SD_ACROSS_DIMENSIONS_CONVERSATION_MODEL,
            share_sd_o_across_subjects: bool = SHARE_SD_ACROSS_SUBJECTS_CONVERSATION_MODEL,
            share_sd_o_across_dimensions: bool = SHARE_SD_ACROSS_DIMENSIONS_CONVERSATION_MODEL,
            coordination_samples: Optional[ModuleSamples] = None):

        coordination = SigmoidGaussianCoordination(
            pymc_model=pymc_model,
            sd_mean_uc0=sd_mean_uc0,
            sd_sd_uc=sd_sd_uc
        )

        # angular_frequency^2 = spring_constant / mass
        state_space = SerialMassSpringDamperLatentComponent(
            uuid="state_space",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            spring_constant=squared_angular_frequency,
            mass=np.ones(num_subjects),
            dampening_coefficient=dampening_coefficient,
            dt=dt,
            mean_mean_a0=mean_mean_a0,
            sd_mean_a0=sd_mean_a0,
            sd_sd_a=sd_sd_a,
            share_mean_a0_across_subjects=share_mean_a0_across_subjects,
            share_sd_a_across_subjects=share_sd_a_across_subjects,
            share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
            share_sd_a_across_dimensions=share_sd_a_across_dimensions
        )

        observation = SerialGaussianObservation(
            uuid="observation",
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=state_space.dimension_size,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_subjects,
            share_sd_o_across_dimensions=share_sd_o_across_dimensions
        )

        group = ComponentGroup(
            uuid="group",
            pymc_model=pymc_model,
            latent_component=state_space,
            observations=[observation]
        )

        super().__init__(
            name=name,
            pymc_model=pymc_model,
            coordination=coordination,
            component_groups=[group],
            coordination_samples=coordination_samples
        )
