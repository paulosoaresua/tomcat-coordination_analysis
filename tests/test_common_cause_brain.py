from coordination.model.real.brain import BrainModel
from coordination.model.config_bundle.brain import BrainBundle 

def test_brain_model_common_cause():
    bundle = BrainBundle()
    bundle.common_cause = True

    brain = BrainModel(config_bundle=bundle)

    samples = brain.draw_samples(num_series=1)
    print(samples)

if __name__ == "__main__":
    test_brain_model_common_cause()
