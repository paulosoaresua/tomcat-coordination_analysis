from __future__ import annotations
import abc
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt

from coordination.module.parametrization import (Parameter,
                                                 HalfNormalParameterPrior,
                                                 NormalParameterPrior)


class LatentComponent(abc.ABC):
    """
    This class represents a latent system component. A latent system component is directly affected
    by coordination which controls to what extend one the latent component from one subject
    influences the same component in another subject in the future.

    We assume latent components evolve as a Gaussian random walk with mean defined by some blending
    strategy that takes coordination and past latent values from other subjects into consideration.

    A latent component can have a function f that transforms that mean in someway. For instance,
    if coordination manifests in an anti-symmetric way, f can be set to f(x) = -x.
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_dimensions: bool,
                 share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_dimensions: bool,
                 dimension_names: Optional[List[str]] = None):
        """
        Creates a latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param self_dependent: whether the latent variables in the component are tied to the
        past values from the same subject. If False, coordination will blend the previous latent
        value of a different subject with the value of the component at time t = 0 for the current
        subject (the latent component's prior for that subject).
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
        latent component).
        @param sd_sd_aa: std of the hyper-prior of sigma_aa (std of the Gaussian random walk of
        the latent component).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_aa_across_subjects: whether to use the same sigma_aa for all subjects.
        @param share_sd_aa_across_dimensions: whether to use the same sigma_aa for all dimensions.
        @param dimension_names: the names of each dimension of the latent component. If not
        informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        """

        # If a parameter is shared across dimensions, we only have one parameter to infer.
        dim_mean_a0_dimensions = 1 if share_mean_a0_across_dimensions else dimension_size
        dim_sd_aa_dimensions = 1 if share_sd_aa_across_dimensions else dimension_size

        # Check if the values passed as hyperparameter are in agreement with the dimension of the
        # variables that we need to infer. Parameters usually have dimensions: num subjects x
        # dimension size, but that changes depending on the sharing options.
        if share_mean_a0_across_subjects:
            assert (dim_mean_a0_dimensions,) == mean_mean_a0.shape
            assert (dim_mean_a0_dimensions,) == sd_mean_a0.shape
        else:
            assert (num_subjects, dim_mean_a0_dimensions) == mean_mean_a0.shape
            assert (num_subjects, dim_mean_a0_dimensions) == sd_mean_a0.shape

        if share_sd_aa_across_subjects:
            assert (dim_sd_aa_dimensions,) == sd_sd_aa.shape
        else:
            assert (num_subjects, dim_sd_aa_dimensions) == sd_sd_aa.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.self_dependent = self_dependent
        self.share_mean_a0_across_subjects = share_mean_a0_across_subjects
        self.share_mean_a0_across_dimensions = share_mean_a0_across_dimensions
        self.share_sd_aa_across_subjects = share_sd_aa_across_subjects
        self.share_sd_aa_across_dimensions = share_sd_aa_across_dimensions
        self.dimension_names = dimension_names

        self.parameters = LatentComponentParameters(mean_mean_a0=mean_mean_a0,
                                                    sd_mean_a0=sd_mean_a0,
                                                    sd_sd_aa=sd_sd_aa)

    @property
    def parameter_names(self) -> List[str]:
        """
        Gets the names of all the parameters used in the distributions of a latent component.

        @return: a list with the parameter names.
        """
        names = [
            self.mean_a0_name,
            self.sd_aa_name
        ]

        return names

    @property
    def mean_a0_name(self) -> str:
        """
        Gets a unique name for the mean of the latent component at time t = 0.

        @return: the name of the parameter that stores the mean of the latent component at time
        t = 0.
        """
        return f"mean_a0_{self.uuid}"

    @property
    def sd_aa_name(self) -> str:
        """
        Gets a unique name for the standard deviation of the latent component at time t = 0.

        @return: the name of the parameter that stores the standard deviation of the latent
        component at time t = 0.
        """
        return f"sd_aa_{self.uuid}"

    def clear_parameter_values(self):
        """
        Clears the values of all the parameters. Their hyper-priors are preserved.
        """
        self.parameters.clear_values()

    def draw_samples(self,
                     coordination: np.ndarray,
                     seed: Optional[int],
                     **kwargs) -> LatentComponentSamples:
        """
        Draws latent component samples using ancestral sampling and some blending strategy with
        coordination and different subjects. This method must be implemented by concrete
        subclasses.

        @param coordination: sampled coordination values.
        @param seed: random seed for reproducibility.
        @param kwargs: extra arguments to be defined by subclasses.

        @return: latent component samples for each coordination series.
        """
        pass

    def _create_initial_mean_variable(self) -> pm.Distribution:
        """
        Creates a latent variable for the mean of the initial state. We assume independence between
        the individual parameters per subject and dimension and sample them from a multivariate
        Gaussian.

        @return: a latent variable with a Gaussian prior.
        """

        dim_mean_a0_dimensions = 1 if self.share_mean_a0_across_dimensions else self.dimension_size

        if self.share_mean_a0_across_subjects:
            # When shared across subjects, only one parameter per dimension is needed.
            mean_a0 = pm.Normal(name=self.mean_a0_name,
                                mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd,
                                size=dim_mean_a0_dimensions,
                                observed=self.parameters.mean_a0.value)
        else:
            # Different parameters per subject and dimension.
            mean_a0 = pm.Normal(name=self.mean_a0_name,
                                mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd,
                                size=(self.num_subjects, dim_mean_a0_dimensions),
                                observed=self.parameters.mean_a0.value)

        return mean_a0

    def _create_transition_standard_deviation_variable(self) -> pm.Distribution:
        """
        Creates a latent variable for the standard deviation of the state transition (Gaussian
        random walk). We assume independence between the individual parameters per subject and
        dimension and sample them from a multivariate Half-Gaussian.

        @return: a latent variable with a Half-Gaussian prior.
        """

        dim_sd_aa_dimensions = 1 if self.share_sd_aa_across_dimensions else self.dimension_size

        if self.share_sd_aa_across_subjects:
            # When shared across subjects, only one parameter per dimension is needed.
            sd_aa = pm.HalfNormal(name=self.sd_aa_name,
                                  sigma=self.parameters.sd_aa.prior.sd,
                                  size=dim_sd_aa_dimensions,
                                  observed=self.parameters.sd_aa.value)
        else:
            # Different parameters per subject and dimension.
            sd_aa = pm.HalfNormal(name=self.sd_aa_name,
                                  sigma=self.parameters.sd_aa.prior.sd,
                                  size=(self.num_subjects, dim_sd_aa_dimensions),
                                  observed=self.parameters.sd_aa.value)

        return sd_aa

    def update_pymc_model(
            self,
            coordination: pm.Distribution,
            dimension_names: List[str],
            observed_values: Optional[TensorTypes] = None,
            mean_a0: Optional[TensorTypes] = None,
            sd_aa: Optional[TensorTypes] = None,
            **kwargs) -> Tuple[
        Union[TensorTypes, pm.Distribution], ...]:

        logp_fn = self._get_logp_fn()
        random_fn = self._get_random_fn()
        latent_component = pm.DensityDist(self.uuid,
                                          *self._get_logp_params(),
                                          logp=logp_fn,
                                          random=random_fn,
                                          dims=dimension_names,
                                          observed=observed_values)

        return latent_component

    def _get_logp_params(self) -> Any:
        """
        Gets parameters to be passed to the logp and random functions.

        @return: a list of unspecified parameters.
        """
        pass

    def _get_logp_fn(self) -> Callable:
        """
        Gets a reference to a logp function.

        @return: a reference to a logp function.
        """
        pass

    def _get_random_fn(self) -> Callable:
        """
        Gets a reference to a random function for prior predictive checks.

        @return: a reference to a random function.
        """
        pass

    def _check_parameter_dimensionality_consistency(self):
        """
        Check if their dimensionality is consistent with the sharing options.
        """

        if self.parameters.mean_a0.value is None:
            raise ValueError("Initial mean parameter value is undefined.")

        if self.parameters.sd_aa.value is None:
            raise ValueError("Transition standard deviation parameter value is undefined.")

        dim_mean_a0_dimensions = 1 if self.share_mean_a0_across_dimensions else self.dimension_size
        if self.share_mean_a0_across_subjects:
            assert (dim_mean_a0_dimensions,) == self.parameters.mean_a0.value.shape
        else:
            assert (self.num_subjects,
                    dim_mean_a0_dimensions) == self.parameters.mean_a0.value.shape

        dim_sd_aa_dimensions = 1 if self.share_sd_aa_across_dimensions else self.dimension_size
        if self.share_sd_aa_across_subjects:
            assert (dim_sd_aa_dimensions,) == self.parameters.sd_aa.value.shape
        else:
            assert (self.num_subjects, dim_sd_aa_dimensions) == self.parameters.sd_aa.value.shape


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class LatentComponentParameters:
    """
    This class stores values and hyper-priors of the parameters of a latent component.
    """

    def __init__(self, mean_mean_a0: np.ndarray, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray):
        """
        Creates an object to store latent component parameter info.

        @param mean_mean_a0: mean of the hyper-prior of the mean at time t = 0.
        @param sd_mean_a0: standard deviation of the hyper-prior of the mean at time t = 0.
        @param sd_sd_aa: standard deviation of the hyper-prior of the standard deviation used in
        the Gaussian random walk when transitioning from one time to the next.
        """
        self.mean_a0 = Parameter(NormalParameterPrior(mean_mean_a0, sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))

    def clear_values(self):
        """
        Set values of the parameters to None. Parameters with None value will be fit to the data
        along with other latent values in the model.
        """
        self.mean_a0.value = None
        self.sd_aa.value = None


class LatentComponentSamples:
    """
    This class stores samples generated by a latent component.
    """

    def __init__(self,
                 values: Union[List[np.ndarray], np.ndarray],
                 time_steps_in_coordination_scale: Union[List[np.ndarray], np.ndarray]):
        """
        Creates an object to store samples.

        @param values: sampled values of the latent component. For serial components, this will be
        a list of time series of values of different sizes. For non-serial components, this will be
        a tensor as the number of observations in time do not change for different sampled time
        series.
        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different time scale from the time scale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the latent component. For serial components, this will be a list of time series of
        indices of different sizes. For non-serial components, this will be a tensor as the number
        of observations in time do not change for different sampled time series.
        """

        self.values = values
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

    @property
    def num_time_steps(self) -> Union[int, np.array]:
        """
        Gets the number of time steps
        @return:
        """

        if isinstance(self.values, List):
            # For a list of sampled series, they can have a different number of time steps. If
            # a scalar is returned, otherwise an array is returned with the number of time steps in
            # each individual series.
            sizes = np.array([sampled_series.shape[-1] for sampled_series in self.values])
            if len(sizes) == 0:
                return 0
            elif len(sizes) == 1:
                return sizes[0]
            else:
                return sizes
        else:
            return self.values.shape[-1]


TensorTypes: Union[np.array, ptt.TensorVariable, ptt.TensorConstant]
