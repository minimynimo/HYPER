#!/bin/bash

cores=1
start_core=12 # Start core (0-indexed)
###

nseeds=1
firstseed=300

nsplits=12

log_name="master_log_LSTM_PUB_kfold.log"

loc="JP"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=-1

# Set default model to "lstm" if not provided
model=${1:-lstm}

output_dir="bash_logs"


mkdir -p "$output_dir" # Create logs directory if it doesn't exist

> "$output_dir/$log_name" # Clear the log file

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do

  python3 hyper/91_LSTM_PUB/main_kfold.py --n_splits=$nsplits --seed=$seed create_splits >> "$output_dir/$log_name" 2>&1
  wait

  for ((split = 0 ; split < $nsplits ; split++ )); do  

    gpucount=$(($gpucount + 1))
    core=$((start_core + (gpucount % cores))) # Ensure core is within start_core to end_core range
    if [ $core -gt $end_core ]; then
      core=$start_core # Wrap around to start_core if exceeding end_core
    fi
    gpu=$((gpucount % 3)) # Assign GPU in a round-robin fashion
    echo $seed $gpucount $core $gpu >> "$output_dir/$log_name"

    if [ "$model" = "lstm" ]; then

      outfile="hyper/out/JP/LSTM_PUB_kfold/reports/pub_lstm.$seed.$split.out"
      start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
      echo "Start Time: $start_time | Seed: $seed | Split: $split" >> "$output_dir/$log_name"

      echo "Running: taskset -c $core python3 hyper/91_LSTM_PUB/main_kfold.py train --gpu=$gpu --no_static=False --concat_static=True --split=$split --split_file=hyper/out/$loc/LSTM_PUB_kfold/data/kfold_splits_seed$seed.p" >> "$output_dir/$log_name"

      taskset -c $core python3 hyper/91_LSTM_PUB/main_kfold.py --gpu=$gpu --no_static=False --concat_static=True --split=$split --split_file="hyper/out/$loc/LSTM_PUB_kfold/data/kfold_splits_seed$seed.p" train > $outfile 2>> "$output_dir/$log_name" &

      wait # Ensure the process completes before logging end time
      end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
      echo "End Time: $end_time | Seed: $seed | Split: $split" >> "$output_dir/$log_name"

    else
      echo "Error: Unsupported model choice '$model'. Supported models: lstm" >> "$output_dir/$log_name"
      exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]
    then
      wait
    fi

  done
done

