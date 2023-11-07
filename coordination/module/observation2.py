from __future__ import annotations
import abc
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm

from coordination.common.types import TensorTypes
from coordination.module.parametrization import (Parameter,
                                                 HalfNormalParameterPrior)


class Observation(abc.ABC):
    """
    This class represents an observation (O) from a latent system component (A). Observations are
    evidence to the model. This implementation samples observation from a Gaussian distribution
     centered on some transformation, g(.),  of the latent components, i.e., O ~ N(g(A), var_o).
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 sd_sd_o: np.ndarray,
                 share_sd_o_across_subjects: bool,
                 share_sd_o_across_dimensions: bool,
                 dimension_names: Optional[List[str]] = None):
        """
        Creates an observation.

        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param sd_sd_o: std of the hyper-prior of sigma_o (std of the Gaussian emission
            distribution).
        @param share_sd_o_across_subjects: whether to use the same sigma_o for all subjects.
        @param share_sd_o_across_dimensions: whether to use the same sigma_o for all dimensions.
        @param dimension_names: the names of each dimension of the observation. If not
            informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        """

        # If a parameter is shared across dimensions, we only have one parameter to infer.
        dim_sd_o_dimensions = 1 if share_sd_o_across_dimensions else dimension_size

        # Check if the values passed as hyperparameter are in agreement with the dimension of the
        # variables that we need to infer. Parameters usually have dimensions: num subjects x
        # dimension size, but that changes depending on the sharing options.
        if share_sd_o_across_subjects:
            assert (dim_sd_o_dimensions,) == sd_sd_o.shape
        else:
            assert (num_subjects, dim_sd_o_dimensions) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dimension_size = dimension_size
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_dimensions = share_sd_o_across_dimensions
        self.dimension_names = dimension_names

        self.parameters = ObservationParameters(sd_sd_o=sd_sd_o)

    @property
    def parameter_names(self) -> List[str]:
        """
        Gets the names of all the parameters used in the distributions of an observation.

        @return: a list with the parameter names.
        """
        names = [
            self.sd_o_name
        ]

        return names

    @property
    def sd_o_name(self) -> str:
        """
        Gets a unique name for the standard deviation of the emission distribution.

        @return: the name of the parameter that stores the standard deviation of emission
            distribution.
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
                     **kwargs) -> ObservationSamples:
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
            observed_values: Optional[TensorTypes] = None,
            mean_a0: Optional[pm.Distribution] = None,
            sd_aa: Optional[pm.Distribution] = None,
            **kwargs) -> Tuple[
        Union[TensorTypes, pm.Distribution], ...]:
        """
        Creates parameters and latent component variables in a PyMC model.

        @param coordination: latent random variable representing a time series of coordination.
        @param observed_values: latent component values if one wants to fix them. This will treat
        the latent component as known and constant. This is not the value of an observation
        component, but the latent component itself.
        @param mean_a0: initial mean of the latent component if previously defined outside of the
        component. This is useful if one wants to share this across different components.
        @param sd_aa: standard deviation of the latent component Gaussian transition distribution
        if previously defined outside of the component. This is useful if one wants to share this
        across different components.
        @param kwargs: extra parameters to be used by child classes.

        @return: random variable representing the latent component series.
        """

        logp_params = self._get_logp_params(coordination=coordination,
                                            observed_values=observed_values,
                                            mean_a0=mean_a0,
                                            sd_aa=sd_aa,
                                            **kwargs)

        dimension_axis_name = f"{self.uuid}_dimension"
        time_axis_name = f"{self.uuid}_time"

        logp_fn = self._get_logp_fn()
        random_fn = self._get_random_fn()
        latent_component = pm.DensityDist(self.uuid,
                                          *logp_params,
                                          logp=logp_fn,
                                          random=random_fn,
                                          dims=[dimension_axis_name, time_axis_name],
                                          observed=observed_values)

        return latent_component

    def _get_logp_params(self,
                         coordination: pm.Distribution,
                         observed_values: Optional[TensorTypes] = None,
                         mean_a0: Optional[TensorTypes] = None,
                         sd_aa: Optional[TensorTypes] = None,
                         **kwargs) -> Any:
        """
        Gets parameters to be passed to the logp and random functions.

        @param coordination: latent random variable representing a time series of coordination.
        @param observed_values: latent component values if one wants to fix them. This will treat
        the latent component as known and constant. This is not the value of an observation
        component, but the latent component itself.
        @param mean_a0: initial mean of the latent component if previously defined outside of the
        component. This is useful if one wants to share this across different components.
        @param sd_aa: standard deviation of the latent component Gaussian transition distribution
        if previously defined outside of the component. This is useful if one wants to share this
        across different components.
        @param kwargs: extra parameters to be used by child classes.

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


class ObservationParameters:
    """
    This class stores values and hyper-priors of the parameters of an observation.
    """

    def __init__(self, sd_sd_o: np.ndarray):
        """
        Creates an object to store observation parameter info.

        @param sd_sd_o: standard deviation of the hyper-prior of the standard deviation used in
            the Gaussian emission distribution.
        """
        self.sd_o = Parameter(HalfNormalParameterPrior(sd_sd_o))

    def clear_values(self):
        """
        Set values of the parameters to None. Parameters with None value will be fit to the data
        along with other latent values in the model.
        """
        self.sd_o.value = None


class ObservationSamples:
    """
    This class stores samples generated by an observation.
    """

    def __init__(self,
                 values: Union[List[np.ndarray], np.ndarray]):
        """
        Creates an object to store samples.

        @param values: sampled values of the observation. For serial observations, this will be
        a list of time series of values of different sizes. For non-serial observations, this will
        be a tensor as the number of observations in time do not change for different sampled time
        series.
        """

        self.values = values

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
