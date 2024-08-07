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
