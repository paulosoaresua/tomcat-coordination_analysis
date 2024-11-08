''' First to check the shape
# import numpy as np
# import pymc as pm
# from coordination.module.coordination.sigmoid_gaussian_coordination import SigmoidGaussianCoordination

# def main():
#     num_series = 3
#     num_time_steps = 10
#     mean_uc0 = 0.0
#     sd_uc = 1.0
#     include_common_cause_options = [False, True]

#     for include_common_cause in include_common_cause_options:
#         print(f"\nTesting with include_common_cause set to {include_common_cause}:")
#         with pm.Model() as model:
#             coordination_module = SigmoidGaussianCoordination(
#                 pymc_model=model,
#                 num_time_steps=num_time_steps,
#                 mean_uc0=mean_uc0,
#                 sd_uc=sd_uc,
#                 include_common_cause=include_common_cause
#             )
#             samples = coordination_module.draw_samples(seed=42, num_series=num_series)
#             expected_shape = (num_series, 3, num_time_steps) if include_common_cause else (num_series, num_time_steps)
#             actual_shape = samples.unbounded_coordination.shape
#             print(expected_shape)
#             print(actual_shape)

# if __name__ == '__main__':
#     main()
'''

'''Second to plot coordinations with only one series
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from coordination.module.coordination.sigmoid_gaussian_coordination import SigmoidGaussianCoordination

def main():
    num_series = 3
    num_time_steps = 50 
    mean_uc0 = 0.0
    sd_uc = 0.1
    include_common_cause_options = [True]

    for include_common_cause in include_common_cause_options:
        print(f"\nTesting with include_common_cause set to {include_common_cause}:")
        with pm.Model() as model:
            coordination_module = SigmoidGaussianCoordination(
                pymc_model=model,
                num_time_steps=num_time_steps,
                mean_uc0=mean_uc0,
                sd_uc=sd_uc,
                include_common_cause=include_common_cause
            )
            samples = coordination_module.draw_samples(seed=42, num_series=num_series)
            plot_samples(samples, include_common_cause, 0)

def plot_samples(samples, include_common_cause, series_index):
    time_steps = np.arange(samples.unbounded_coordination.shape[-1])
    plt.figure(figsize=(10, 4))
    
    if include_common_cause:
        for j in range(3):
            plt.plot(time_steps, samples.unbounded_coordination[series_index, j, :], label=f'Coordination {j+1}')
    else:
        plt.plot(time_steps, samples.unbounded_coordination[series_index, :], label='Coordination')

    plt.title(f'Series {series_index + 1} Coordination Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Coordination')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
'''

import numpy as np
import pymc as pm
from coordination.module.coordination.sigmoid_gaussian_coordination_3d import SigmoidGaussianCoordination3D
from coordination.module.coordination.dirichlet_gaussian_coordination_3d import DirichletGaussianCoordination3D

def main():
    num_series = 3
    num_time_steps = 10
    mean_uc0 = 0.0
    sd_uc = 1.0
    include_common_cause_options = [True]

    for include_common_cause in include_common_cause_options:
        print(f"\nTesting with include_common_cause set to {include_common_cause}:")
        with pm.Model() as model:
            coordination_module = DirichletGaussianCoordination3D(
                pymc_model=model,
                num_time_steps=num_time_steps,
                mean_uc0_individualism=mean_uc0,
                sd_uc_individualism=sd_uc,
                mean_uc0_coordination=mean_uc0,
                sd_uc_coordination=sd_uc,
                mean_uc0_common_cause=mean_uc0,
                sd_uc_common_cause=sd_uc
            )
            samples = coordination_module.draw_samples(seed=42, num_series=num_series)
            expected_shape = (num_series, 3, num_time_steps)
            actual_shape = samples.unbounded_coordination.shape
            print(expected_shape)
            print(actual_shape)


            coordination_module.create_random_variables()

            mean_uc0_i_rv = coordination_module.mean_uc0_individualism_random_variable
            sd_uc_i_rv = coordination_module.sd_uc_individualism_random_variable
            mean_uc0_c_rv = coordination_module.mean_uc0_coordination_random_variable
            sd_uc_c_rv = coordination_module.sd_uc_coordination_random_variable
            mean_uc0_cc_rv = coordination_module.mean_uc0_common_cause_random_variable
            sd_uc_cc_rv = coordination_module.sd_uc_common_cause_random_variable
            unbounded_coordination_rv = model.named_vars['unbounded_coordination']
            coordination_rv = coordination_module.coordination_random_variable

            mean_uc0_samples = pm.draw(mean_uc0_i_rv, draws=3)
            print(mean_uc0_samples)
            sd_uc_samples = pm.draw(sd_uc_i_rv, draws=3)
            print(sd_uc_samples)
            mean_uc0_samples = pm.draw(mean_uc0_c_rv, draws=3)
            print(mean_uc0_samples)
            sd_uc_samples = pm.draw(sd_uc_c_rv, draws=3)
            print(sd_uc_samples)
            mean_uc0_samples = pm.draw(mean_uc0_cc_rv, draws=3)
            print(mean_uc0_samples)
            sd_uc_samples = pm.draw(sd_uc_cc_rv, draws=3)
            print(sd_uc_samples)
            unbounded_coordination_samples = pm.draw(unbounded_coordination_rv, draws=3)
            print(unbounded_coordination_samples.shape)
            print(unbounded_coordination_samples)
            coordination_samples = pm.draw(coordination_rv, draws=3)
            print(coordination_samples.shape)
            print(coordination_samples)

if __name__ == '__main__':
    main()