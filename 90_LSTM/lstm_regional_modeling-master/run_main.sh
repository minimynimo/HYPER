#!/bin/bash

cores=1
start_core=20 # Start core (0-indexed)
###

nseeds=1
firstseed=300

static_bool=false


if [ "$static_bool" = true ]; then
    log_name="master_log_LSTM_static.log"
else
    log_name="master_log_LSTM_nostatic_MAIN.log"
fi


loc="JP"

end_core=$((start_core + cores - 1))  # Calculate end core based on core count

gpucount=1

# Set default model to "lstm" if not provided
model=${1:-lstm}

output_dir="/data0/funato/99_work/bash_logs"

mkdir -p "$output_dir" # Create logs directory if it doesn't exist

> "$output_dir/$log_name" # Clear the log file


for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    gpucount=$(($gpucount + 1))
    core=$((start_core + (gpucount % cores))) # Ensure core is within start_core to end_core range
    if [ $core -gt $end_core ]; then
        core=$start_core # Wrap around to start_core if exceeding end_core
    fi
    gpu=$((gpucount % 3)) # Assign GPU in a round-robin fashion
    echo $seed $gpucount $core $gpu >> "$output_dir/$log_name"

    if [ "$model" = "lstm" ]; then
        if [ "$static_bool" = true ]; then
            #add current date and time to the outfile name
            outfile="out/$loc/LSTM/reports_static/lstm.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        else
            outfile="out/$loc/LSTM/reports_nostatic/lstm.$seed.$(date +%Y-%m-%d_%H-%M-%S).out"
        fi

        mkdir -p "$(dirname "$outfile")" # Create output directory if it doesn't exist
        start_time=$(date '+%Y-%m-%d %H:%M:%S') # Log start time1
        echo "Start Time: $start_time | Seed: $seed" >> "$output_dir/$log_name"
        
        if [ "$static_bool" = true ]; then  ## static data is used
            echo "Running: taskset -c $core /data0/funato/.venv/bin/python /data0/funato/99_work/91_LSTM/lstm_regional_modeling-master/main.py train --no_static=False --concat_static=False --num_workers=$cores" >> "$output_dir/$log_name"
            taskset -c $core /data0/funato/.venv/bin/python /data0/funato/99_work/91_LSTM/lstm_regional_modeling-master/main.py train --no_static=False --concat_static=False --num_workers=$cores > $outfile 2>> "$output_dir/$log_name" &
        else  ## no static data is used
            echo "Running: taskset -c $core /data0/funato/.venv/bin/python /data0/funato/99_work/91_LSTM/lstm_regional_modeling-master/main.py train --no_static=True --concat_static=False --num_workers=$cores" >> "$output_dir/$log_name"
            taskset -c $core /data0/funato/.venv/bin/python /data0/funato/99_work/91_LSTM/lstm_regional_modeling-master/main.py train --no_static=True --concat_static=False --num_workers=$cores > $outfile 2>> "$output_dir/$log_name" &
        fi

        wait # Ensure the process completes before logging end time
        end_time=$(date '+%Y-%m-%d %H:%M:%S') # Log end time
        echo "End Time: $end_time | Seed: $seed" >> "$output_dir/$log_name"
    else
        echo "Error: Unsupported model choice '$model'. Supported models: lstm" >> "$output_dir/$log_name"
        exit 1
    fi

    if [ $((gpucount % cores)) -eq $((cores - 1)) ]; then
        wait
    fi
done

###RUN USING####
# chmod +x /data0/funato/99_work/91_LSTM/lstm_regional_modeling-master/run_main.sh
# nohup /data0/funato/99_work/91_LSTM/lstm_regional_modeling-master/run_main.sh > /dev/null 2>&1 &
