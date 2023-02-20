#!/bin/bash

parallel_inference_script_dir="$PROJECT_DIR/scripts"

export PYTHONPATH="$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"

pushd "$parallel_inference_script_dir" > /dev/null || exit 1
  python3 inference.py \
    --out_dir="$OUT_DIR" \
    --experiment_ids="$EXPERIMENT_IDS" \
    --evidence_filepath="$EVIDENCE_FILEPATH" \
    --model="$MODEL" \
    --burn_in="$BURN_IN" \
    --num_samples="$NUM_SAMPLES" \
    --num_chains="$NUM_CHAIN" \
    --seed="$SEED" \
    --num_inference_jobs="$NUM_INFERENCE_JOBS" \
    --do_prior="$DO_PRIOR" \
    --do_posterior="$DO_POSTERIOR" \
    --initial_coordination="$INITIAL_COORDINATION" \
    --num_subjects="$NUM_SUBJECTS" \
    --brain_channels="$BRAIN_CHANNELS" \
    --vocalic_features="$VOCALIC_FEATURES" \
    --self_dependent="$SELF_DEPENDENT" \
    --sd_uc="$SD_UC" \
    --sd_mean_a0_brain="$SD_MEAN_A0_BRAIN" \
    --sd_sd_aa_brain="$SD_SD_AA_BRAIN" \
    --sd_sd_o_brain="$SD_SD_O_BRAIN" \
    --sd_mean_a0_body="$SD_MEAN_A0_BODY" \
    --sd_sd_aa_body="$SD_SD_AA_BODY" \
    --sd_sd_o_body="$SD_SD_O_BODY" \
    --a_mixture_weights="$A_MIXTURE_WEIGHTS" \
    --sd_mean_a0_vocalic="$SD_MEAN_A0_VOCALIC" \
    --sd_sd_aa_vocalic="$SD_SD_AA_VOCALIC" \
    --sd_sd_o_vocalic="$SD_SD_O_VOCALIC" \
    --a_p_semantic_link="$A_P_SEMANTIC_LINK" \
    --b_p_semantic_link="$B_P_SEMANTIC_LINK"
popd > /dev/null || exit 1

echo "Press any key to continue..."
read -r -n 1

exit 0



