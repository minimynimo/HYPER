"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import glob
import os
import pickle
import sys
import time

import pandas as pd

# number of ensemble members
nSeeds = 5
firstSeed = 300
## CHANGE #### ALSO CHANGE main_region.py, weight_file in main_region.py

loc = "JP"

region_list = ['Hokkaido', 'Tohoku', 'North Central', 'South Central', 'West']
region = region_list[4]

# user inputs
#experiment = sys.argv[1]
experiment="pub_lstm"
#gpu = sys.argv[2]
gpu = 1

seed_list = range(firstSeed, firstSeed + nSeeds)  # loop through randomized ensemble

log_file = f"hyper/out/{loc}/LSTM_PUB_region/LSTM_PUB_eval_log.txt"
with open(log_file, 'w') as log:
    log.write("Seed,Start Time,End Time,Elapsed Time (seconds)\n")

# Create a log file for evaluation
log_dir = f"hyper/out/{loc}/LSTM_PUB_region/logs"
os.makedirs(log_dir, exist_ok=True)
eval_log_file = f"{log_dir}/lstm_pub_eval_{region}.log"
with open(eval_log_file, 'w') as eval_log:
    eval_log.write("Evaluation Log\n")
    eval_log.flush()

# Initialize ensemble dictionaries
ens_dict_cal = {}
ens_dict_val = {}

# This loop will run the evaluation procedure for all splits of all PUB ensembles
for seed in seed_list:  # loop through randomized ensemble
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

    # get the correct run directory by reading the screen report
    fname = f"hyper/out/{loc}/LSTM_PUB_region/reports/{experiment}.{region}.{seed}.out"
    with open(eval_log_file, 'a') as eval_log:
        eval_log.write(f"Working on seed: {seed} -- file: {fname}\n")
        eval_log.flush()
    # Open with error handling for encoding issues
    with open(fname, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    split_file = lines[9].split(': ')[1][:-1]
    run_dir = lines[28].split('attributes in ')[1].split('attributes')[0]
    print("run_dir", run_dir)

    weight_file = f"{run_dir}/model_epoch30.pt"
    if not os.path.exists(weight_file):
        with open(eval_log_file, 'a') as eval_log:
            eval_log.write(f"Warning: {weight_file} does not exist.\n")
            eval_log.flush()
        # skip this iteration for both seed and ensemble calculations

    run_command = f"python3 hyper/91_LSTM_PUB/main_region.py --gpu={gpu} --run_dir={run_dir} --split_file={split_file} --region={region}  evaluate"
    os.system(run_command)

    # grab the test output file for this split
    file_seed = run_dir.split('seed')[1][:-1]
    print("file_seed", file_seed)
    try:
        results_file_cal = glob.glob(f"{run_dir}/*lstm*seed{file_seed}_cal.p")[0]
        results_file_val = glob.glob(f"{run_dir}/*lstm*seed{file_seed}_val.p")[0]
    except IndexError:
        print(f"Error: Results files for {region} seed {seed} are missing. Skipping.")
        continue

    with open(results_file_cal, 'rb') as f:
        partial_dict_cal = pickle.load(f)
    with open(results_file_val, 'rb') as f:
        partial_dict_val = pickle.load(f)

    # store in a dictionary for this seed
    seed_dict_cal = partial_dict_cal
    seed_dict_val = partial_dict_val

    # screen report
    print(seed, len(seed_dict_cal), len(seed_dict_val))

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_time = end_time - start_time

    # Log the timing information
    with open(log_file, 'a') as log:
        log.write(f"{seed},{start_time_str},{end_time_str},{elapsed_time:.2f}\n")
        log.flush()

    with open(eval_log_file, 'a') as eval_log:
        eval_log.write(f"Seed {seed} completed. Start: {start_time_str}, End: {end_time_str}, Elapsed: {elapsed_time:.2f} seconds\n")
        eval_log.flush()

    # --- end of split loop -----------------------------------------

    # create the ensemble dictionary
    for basin in seed_dict_cal: 
        # rename the columns to include the seed number
        seed_dict_cal[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
        # append to CSV for calibration
        output_dir = f"hyper/out/{loc}/LSTM_PUB_region/ensemble/{region}/test_basin/predict"
        os.makedirs(output_dir, exist_ok=True)
        fname_cal = f"{output_dir}/{experiment}_{basin}_cal.csv"
        if not os.path.exists(fname_cal):
            # Create a new CSV with dates as a column and the first seed's predictions as a row
            seed_dict_cal[basin].T.to_csv(fname_cal, header=True, index_label="Date")
        else:
            # Append the new seed's predictions as a new row
            seed_dict_cal[basin].T.to_csv(fname_cal, mode='a', header=False, index_label="Date")

        # Update ensemble dictionary
        if basin not in ens_dict_cal:
            ens_dict_cal[basin] = seed_dict_cal[basin]
        else:
            ens_dict_cal[basin] = pd.merge(
                ens_dict_cal[basin],
                seed_dict_cal[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True
            )

    for basin in seed_dict_val:
        seed_dict_val[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
        # append to CSV for validation
        fname_val = f"{output_dir}/{experiment}_{basin}_val.csv"
        if not os.path.exists(fname_val):
            # Create a new CSV with dates as a column and the first seed's predictions as a row
            seed_dict_val[basin].T.to_csv(fname_val, header=True, index_label="Date")
        else:
            # Append the new seed's predictions as a new row
            seed_dict_val[basin].T.to_csv(fname_val, mode='a', header=False, index_label="Date")

        # Update ensemble dictionary
        if basin not in ens_dict_val:
            ens_dict_val[basin] = seed_dict_val[basin]
        else:
            ens_dict_val[basin] = pd.merge(
                ens_dict_val[basin],
                seed_dict_val[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True
            )

# --- end of seed loop -----------------------------------------

# calculate ensemble mean
ensemble_start_time = time.time()
ensemble_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ensemble_start_time))

for basin in ens_dict_cal:
    simdf = ens_dict_cal[basin].filter(regex='qsim_')
    ensMean = simdf.mean(axis=1)
    ens_dict_cal[basin].insert(0, 'qsim', ensMean)

for basin in ens_dict_val:
    simdf = ens_dict_val[basin].filter(regex='qsim_')
    ensMean = simdf.mean(axis=1)
    ens_dict_val[basin].insert(0, 'qsim', ensMean)

ensemble_end_time = time.time()
ensemble_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ensemble_end_time))
ensemble_elapsed_time = ensemble_end_time - ensemble_start_time

# Log the ensemble calculation timing information
with open(log_file, 'a') as log:
    log.write(f"Ensemble Calculation,{ensemble_start_time_str},{ensemble_end_time_str},{ensemble_elapsed_time:.2f}\n")

with open(eval_log_file, 'a') as eval_log:
    eval_log.write(f"Ensemble calculation completed. Start: {ensemble_start_time_str}, End: {ensemble_end_time_str}, Elapsed: {ensemble_elapsed_time:.2f} seconds\n")
    eval_log.flush()

print(f"Ensemble calculation completed. Start: {ensemble_start_time_str}, End: {ensemble_end_time_str}, Elapsed: {ensemble_elapsed_time:.2f} seconds")
