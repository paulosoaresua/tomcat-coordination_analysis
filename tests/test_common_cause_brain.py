from coordination.model.real.brain import BrainModel
from coordination.model.config_bundle.brain import BrainBundle 

def test_brain_model_common_cause():
    bundle = BrainBundle()
    bundle.common_cause = True
    
    # Set required parameters for the model
    bundle.mean_uc0 = 0
    bundle.sd_uc = 1
    bundle.fnirs_mean_a0 = 0
    bundle.fnirs_sd_a = 1
    bundle.fnirs_sd_o = 1
    bundle.gsr_mean_a0 = 0
    bundle.gsr_sd_a = 1
    bundle.gsr_sd_o = 1
    bundle.vocalic_mean_a0 = 0
    bundle.vocalic_sd_a = 1
    bundle.vocalic_sd_o = 1
    bundle.semantic_link_sd_s = 1
    
    brain = BrainModel(config_bundle=bundle)

    samples = brain.draw_samples(num_series=1)
    print(samples)

if __name__ == "__main__":
    test_brain_model_common_cause()
"""
File "/Users/mlyang721/miniconda3/envs/coordination/lib/python3.11/site-packages/scipy/stats/_distn_infrastructure.py", line 1028, in rvs
    args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 6, in _parse_args_rvs
  File "/Users/mlyang721/miniconda3/envs/coordination/lib/python3.11/site-packages/scipy/stats/_distn_infrastructure.py", line 909, in _argcheck_rvs
    raise ValueError("size does not match the broadcast shape of "
ValueError: size does not match the broadcast shape of the parameters. (1, 1, 2), (1, 1, 2), (1, 3, 1)
"""

