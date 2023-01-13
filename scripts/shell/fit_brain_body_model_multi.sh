#!/bin/bash

set -u

__help() {
  echo
  echo "This script infers coordination in a Brain Body model."
  echo
  echo
  echo "One of three conditions must be informed when calling this script:"
  echo
  echo "Syntax: train [-h|abcj:n:] <condition>"
  echo "options:"
  echo "h                        Print help."
  echo "a                        Estimates coordination on mission 1 "
  echo "b                        Estimates coordination on mission 2."
  echo "c                        Estimates coordination on both missions."
  echo "g                        Whether to use a model that considers the speakers' gender."
  echo "l                        Whether to use a model that considers speech semantic link."
  echo "d                        Number of particles (default 30000)."
  echo "r                        Reference date (e.g. 2022.12.02--15)."
  echo "t                        Training type (e.g. single_execution_no_self_dep)."
  echo "j <number of processes>  For parallel execution."
  echo "f <features>             Vocalic features used during training."
  echo "n <condition>            One of the following values."
  echo "   no     - No Advisor"
  echo "   human  - Human Advisor"
  echo "   tomcat - ToMCAT Advisor"
  echo "   all    - All Conditions"
  echo
}

--brain_data="/Users/paulosoares/data/IJCAI_2023/ping_pong/parsed/1_hz/ping_pong_cooperative_0/brain_signals.csv"
--body_data="/Users/paulosoares/data/IJCAI_2023/ping_pong/parsed/1_hz/ping_pong_cooperative_0/body_movements.csv"
--n_time_steps=120
--c0=0.5
--burn_in=1000
--n_samples=1000
--n_chains=2
--n_i_jobs=4
--n_t_jobs=1
--seed=0
--out_dir="/Users/paulosoares/data/IJCAI_2023/ping_pong/inference"

data_idx=-1
do_mission1=0
do_mission2=0
do_both_missions=0
num_jobs=1
condition_value=""
num_particles=30000
features="pitch, intensity"
gendered=0
link=0
while getopts ":habcglp:r:t:j:f:n:" option; do
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
    g) # gendered
      gendered=1
      ;;
    l) # speech semantic link
      link=1
      ;;
    p) # number of particles
      num_particles=$OPTARG
      ;;
    r) # reference date
      ref_date=$OPTARG
      ;;
    t) # training type
      training_type=$OPTARG
      ;;
    j) # number of processes
      num_jobs=$OPTARG
      ;;
    f) # features
      features=$OPTARG
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
  condition="all_conditions"
else
  echo "Condition invalid. Please provide a valid condition (call this script with the option -h to see the list)."
  exit 1
fi

# Export path to the project so it recognize the custom packages.
export PYTHONPATH="$HOME/code/tomcat-coordination"

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
                     --features="$features" \
                     --gendered=$gendered \
                     --link=$link

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
                     --features="$features" \
                     --gendered=$gendered \
                     --link=$link

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
                       --features="$features" \
                       --gendered=$gendered \
                       --link=$link

    python3 plot_coordination.py --inference_path="$results_dir/all_missions/$ref_date/inference_summaries.pkl" \
                                 --dataset_path="$data_dir/all_missions_dataset.pkl" \
                                 --width=800 \
                                 --out_dir="$plots_dir/all_missions/$ref_date"

    sleep 1 # To close all the semaphores
  fi

popd > /dev/null || exit

echo ""
echo "-----------------------------------------"
echo " DONE INFERRING!!!                        "
echo "-----------------------------------------"
echo ""

exit 0
