#!/bin/bash

set -u

__help() {
  echo
  echo "This script infers coordination based on a pre-trained BetaGaussian model with just INTENSITY as feature."
  echo
  echo "It estimates coordination on data from mission1, mission2 and both missions combined "
  echo "in sequence, if requested via the options below. By default, it runs "
  echo "in a single process."
  echo ""
  echo
  echo "One of three conditions must be informed when calling this script:"
  echo
  echo "Syntax: train [-h|abcj:n:] <condition>"
  echo "options:"
  echo "h                        Print help."
  echo "a                        Estimates coordination on mission 1 "
  echo "b                        Estimates coordination on mission 2."
  echo "c                        Estimates coordination on both missions."
  echo "p                        Number of particles (default 30000)."
  echo "r                        Reference date (e.g. 2022.12.02--15)."
  echo "j <number of processes>  For parallel execution."
  echo "n <condition>            One of the following values."
  echo "   no     - No Advisor"
  echo "   human  - Human Advisor"
  echo "   tomcat - ToMCAT Advisor"
  echo "   all    - All Advisors"
  echo
}

do_mission1=0
do_mission2=0
do_both_missions=0
num_jobs=1
condition_value=""
num_particles=30000
while getopts ":habcp:r:j:n:" option; do
  case $option in
    h) # display Help
      __help
      exit;;
    a) # Mission 1
      do_mission1=1
      ;;
    b) # Mission 2
      do_mission2=1
      ;;
    c) # Mission 1 and 2
      do_both_missions=1
      ;;
    p) # number of particles
      num_particles=$OPTARG
      ;;
    r) # reference date
      ref_date=$OPTARG
      ;;
    j) # number of processes
      num_jobs=$OPTARG
      ;;
    n) # condition
      condition_value=$OPTARG
      ;;
    \?) # Invalid option
      echo "ERROR: invalid option."
      exit;;
  esac
done

if [[ "$condition_value" == "no" ]]; then
  condition="no_advisor"
elif [[ "$condition_value" == "human" ]]; then
  condition="human_advisor"
elif [[ "$condition_value" == "tomcat" ]]; then
  condition="tomcat_advisor"
elif [[ "$condition_value" == "all" ]]; then
  condition="all_advisors"
else
  echo "Condition invalid. Please provide a valid condition (call this script with the option -h to see the list)."
  exit 1
fi

# Export path to the project so it recognize the custom packages.
export PYTHONPATH="$HOME/code/tomcat-coordination"

training_type="single_execution_intensity_only"
data_dir="$HOME/data/study-3_2022/datasets/$condition"
models_dir="$HOME/data/study-3_2022/models/beta_gaussian/$condition/$training_type"
results_dir="$HOME/data/study-3_2022/inferences/beta_gaussian/$condition/$training_type"
plots_dir="$HOME/data/study-3_2022/plots/beta_gaussian/$condition/$training_type"

pushd "$HOME/code/tomcat-coordination/scripts" > /dev/null || exit

  if [[ $do_mission1 -ne 0 ]]; then
    echo ""
    echo "-----------------------------------------"
    echo " MISSION 1                               "
    echo "-----------------------------------------"
    echo ""
    python3 infer.py --dataset_path="$data_dir/mission1_dataset.pkl" \
                     --model_path="$models_dir/mission1/$ref_date/model.pkl" \
                     --n_particles="$num_particles" \
                     --seed=0 \
                     --n_jobs="$num_jobs" \
                     --out_dir="$results_dir/mission1/$ref_date" \
                     --disable_feature=0

    python3 plot_coordination.py --inference_path="$results_dir/mission1/$ref_date/inference_summaries.pkl" \
                                 --dataset_path="$data_dir/mission1_dataset.pkl" \
                                 --width=800 \
                                 --out_dir="$plots_dir/mission1/$ref_date"

    sleep 1 # To close all the semaphores
  fi

  if [[ $do_mission2 -ne 0 ]]; then
    echo ""
    echo "-----------------------------------------"
    echo " MISSION 2                               "
    echo "-----------------------------------------"
    echo ""
    python3 infer.py --dataset_path="$data_dir/mission2_dataset.pkl" \
                     --model_path="$models_dir/mission2/$ref_date/model.pkl" \
                     --n_particles="$num_particles" \
                     --seed=0 \
                     --n_jobs="$num_jobs" \
                     --out_dir="$results_dir/mission2/$ref_date" \
                     --disable_feature=0

    python3 plot_coordination.py --inference_path="$results_dir/mission2/$ref_date/inference_summaries.pkl" \
                                 --dataset_path="$data_dir/mission2_dataset.pkl" \
                                 --width=800 \
                                 --out_dir="$plots_dir/mission2/$ref_date"

    sleep 1 # To close all the semaphores
  fi

  if [[ $do_both_missions -ne 0 ]]; then
    echo ""
    echo "-----------------------------------------"
    echo " ALL MISSIONS                            "
    echo "-----------------------------------------"
    echo ""
    python3 infer.py --dataset_path="$data_dir/all_missions_dataset.pkl" \
                     --model_path="$models_dir/all_missions/$ref_date/model.pkl" \
                     --n_particles="$num_particles" \
                     --seed=0 \
                     --n_jobs="$num_jobs" \
                     --out_dir="$results_dir/all_missions/$ref_date" \
                     --disable_feature=0

    python3 plot_coordination.py --inference_path="$results_dir/all_missions/$ref_date/inference_summaries.pkl" \
                                 --dataset_path="$data_dir/all_missions_dataset.pkl" \
                                 --width=800 \
                                 --out_dir="$plots_dir/all_missions/$ref_date"

    sleep 1 # To close all the semaphores
  fi

popd > /dev/null || exit

echo ""
echo "-----------------------------------------"
echo " DONE TRAINING!!!                        "
echo "-----------------------------------------"
echo ""

exit 0
