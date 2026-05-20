#!/bin/bash

cores=5
start_core=0 # Start core (0-indexed)
###

firstseed=300 #300
nSeeds=5

loc="JP"

regions=('Hokkaido' 'Tohoku' 'North Central' 'South Central' 'West')

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

# Use a numeric list for seed_list, not a string
seed_list=()
for (( seed = firstseed ; seed < firstseed + nSeeds ; seed++ )); do
  seed_list+=($seed)
done

echo "Seed list: ${seed_list[@]}"

gpucount=-1

# Set default model to "lstm" if not provided
model=${1:-lstm}

output_dir="bash_logs"

mkdir -p "$output_dir" # Create logs directory if it doesn't exist

for region in "${regions[@]}"; do
  log_name="master_log_LSTM_PUB_region_$region.log"
  > "$output_dir/$log_name" # Clear the log file

  #for (( seed = $firstseed ; seed < $((firstseed + 100)) ; seed++ )); do  # Loop for 100 seeds
  for seed in "${seed_list[@]}"; do  # Loop seeds in the seed_list
    # Assign core and GPU in a round-robin fashion

    core=$((start_core + (gpucount + 1) % cores))
    gpu=$(((gpucount + 1) % 3)) # Assign GPU in a round-robin fashion
    gpucount=$((gpucount + 1))
    
    echo "Seed: $seed | Core: $core | GPU: $gpu" >> "$output_dir/$log_name"

    # Create region train-test splits in the background
    python3 91_LSTM_PUB/main_region.py --seed=$seed --region=$region create_region_train_test >> "$output_dir/$log_name" 2>&1 &

    # Start training in the background
    if [ "$model" = "lstm" ]; then

      outfile="out/JP/LSTM_PUB/region/reports/pub_lstm.$region.$seed.out"
      start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time
      echo "Start Time: $start_time | Seed: $seed | Region: $region" >> "$output_dir/$log_name"

      echo "Running: taskset -c $core python3 91_LSTM_PUB/main_region.py train --gpu=$gpu --no_static=False --concat_static=True --split_file=out/$loc/LSTM_PUB/region/data/region_splits_${region}_seed${seed}.p --region=$region" >> "$output_dir/$log_name"

      # Execute the training command in the background
      taskset -c $core python3 91_LSTM_PUB/main_region.py --gpu=$gpu --no_static=False --concat_static=True --split_file="out/$loc/LSTM_PUB/region/data/region_splits_${region}_seed${seed}.p" --region=$region train > $outfile 2>> "$output_dir/$log_name" &


    else
      echo "Error: Unsupported model choice '$model'. Supported models: lstm" >> "$output_dir/$log_name"
      exit 1
    fi

    # Limit the number of parallel processes to the number of cores
    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
      wait # Wait for all background processes to complete
    fi

  done

  # Wait for any remaining background processes to complete before moving to the next region
  wait
done
