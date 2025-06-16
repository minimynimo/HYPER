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
nSplits = 12
nSeeds = 10
firstSeed = 300
## CHANGE #### ALSO CHANGE main.py, weight_file in main.py

loc = "JP"

# user inputs
#experiment = sys.argv[1]
experiment="pub_lstm"
#gpu = sys.argv[2]
gpu = 1

log_file = f"hyper/out/{loc}/LSTM_PUB/LSTM_PUB_eval_log.txt"
with open(log_file, 'w') as log:
    log.write("Seed,Split,Start Time,End Time,Elapsed Time (seconds)\n")

# This loop will run the evaluation procedure for all splits of all PUB ensembles
for seed in range(firstSeed, firstSeed + nSeeds):  # loop through randomized ensemble
    for split in range(nSplits):  # number of k-fold splits
        start_time = time.time()
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

        # get the correct run directory by reading the screen report
        fname = f"hyper/out/{loc}/LSTM_PUB/reports/{experiment}.{seed}.{split}.out"
        print(f"Working on seed: {seed} -- file: {fname}")
        f = open(fname)
        lines = f.readlines()

        split_file = lines[11].split(': ')[1][:-1]
        split_num = lines[10].split(' ')[1][:-1]
        run_dir = lines[26].split('attributes in ')[1].split('attributes')[0]
        print("run_dir", run_dir)

        weight_file = f"{run_dir}/model_epoch30.pt"
        if not os.path.exists(weight_file):
            print(f"Warning: {weight_file} does not exist.")

            print(f"Rerunning training for seed {seed}, split {split} to generate missing model file.")
            
            # Construct the output file and log file paths
            outfile = f"{run_dir}/training_output_seed{seed}_split{split}.log"
            log_name = f"LSTM_PUB_train_log_seed{seed}_split{split}.txt"
            output_dir = f"hyper/out/{loc}/LSTM_PUB/logs_redo"
            os.makedirs(output_dir, exist_ok=True)

            # Rerun training command
            rerun_command = (
                f"python3 hyper/91_LSTM_PUB/main.py --gpu={gpu} "
                f"--no_static=False --concat_static=True --split={split} "
                f"--split_file=\"hyper/out/{loc}/LSTM_PUB/data/kfold_splits_seed{seed}.p\" train "
                f"> {outfile} 2>> \"{output_dir}/{log_name}\" &"
            )
            os.system(rerun_command)
            print(f"Training rerun command executed: {rerun_command}")

        run_command = f"python3 hyper/91_LSTM_PUB/main.py --gpu={gpu} --run_dir={run_dir} --split={split_num} --split_file={split_file} evaluate"
        os.system(run_command)

        # grab the test output file for this split
        file_seed = run_dir.split('seed')[1][:-1]
        print("file_seed", file_seed)
        try:
            results_file_cal = glob.glob(f"{run_dir}/*lstm*seed{file_seed}_cal.p")[0]
            results_file_val = glob.glob(f"{run_dir}/*lstm*seed{file_seed}_val.p")[0]
        except IndexError:
            print(f"Error: Results files for seed {seed}, split {split} are missing. Skipping.")
            continue

        with open(results_file_cal, 'rb') as f:
            partial_dict_cal = pickle.load(f)
        with open(results_file_val, 'rb') as f:
            partial_dict_val = pickle.load(f)

        # store in a dictionary for this seed
        if split > 0:
            seed_dict_cal.update(partial_dict_cal)
            seed_dict_val.update(partial_dict_val)
        else:
            seed_dict_cal = partial_dict_cal
            seed_dict_val = partial_dict_val

        # screen report
        print(seed, split, len(seed_dict_cal), len(seed_dict_val))

        end_time = time.time()
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        elapsed_time = end_time - start_time

        # Log the timing information
        with open(log_file, 'a') as log:
            log.write(f"{seed},{split},{start_time_str},{end_time_str},{elapsed_time:.2f}\n")

        print(f"Seed {seed}, Split {split} completed. Start: {start_time_str}, End: {end_time_str}, Elapsed: {elapsed_time:.2f} seconds")

        # --- end of split loop -----------------------------------------

    # create the ensemble dictionary
    for basin in seed_dict_cal: 
        # rename the columns to include the seed number
        seed_dict_cal[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
    if seed == firstSeed:
        ens_dict_cal = seed_dict_cal
    else:
        for basin in seed_dict_cal:
            ens_dict_cal[basin] = pd.merge(
                ens_dict_cal[basin],
                seed_dict_cal[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)

    for basin in seed_dict_val:
        seed_dict_val[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
    if seed == firstSeed:
        ens_dict_val = seed_dict_val
    else:
        for basin in seed_dict_val:
            ens_dict_val[basin] = pd.merge(
                ens_dict_val[basin],
                seed_dict_val[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)
            
    

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
    log.write(f"Ensemble Calculation,,{ensemble_start_time_str},{ensemble_end_time_str},{ensemble_elapsed_time:.2f}\n")

print(f"Ensemble calculation completed. Start: {ensemble_start_time_str}, End: {ensemble_end_time_str}, Elapsed: {ensemble_elapsed_time:.2f} seconds")

# save the ensemble results as CSV files
output_dir = f"hyper/out/{loc}/LSTM_PUB/ensemble"
os.makedirs(output_dir, exist_ok=True)
for basin, df in ens_dict_cal.items():
    fname = f"{output_dir}/{experiment}_{basin}_cal.csv"
    df.to_csv(fname)

for basin, df in ens_dict_val.items():
    fname = f"{output_dir}/{experiment}_{basin}_val.csv"
    df.to_csv(fname)

print(f"Ensemble results saved to CSV files in {output_dir}")
