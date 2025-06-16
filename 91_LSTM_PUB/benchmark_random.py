import pandas as pd
import numpy as np
import os

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 1
loc, ver = "JP", 2

firstSeed = 300
nSeeds = 100
seed_list = list(range(firstSeed, firstSeed + nSeeds))

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

if loc == "JP" and ver == 1:
    file_tot_num  = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    train_basin_int_list = [3,5,10,15,20,30,50,70]
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
varssim_dir = f"hyper/data/MERVJP/varssim_nocal/{ver_name}"

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

file_num_list = list(range(1, file_tot_num + 1))

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

def BMK(obs_data,sim_data,benchmark):    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        obs_std = np.where(obs_std == 0, 1e-6, obs_std)
        obs_ave = np.where(obs_ave == 0, 1e-6, obs_ave) 
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "VE":
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))


for train_basin_int in train_basin_int_list:
    input_dir = f"hyper/out/{loc}/LSTM_PUB_random/ensemble/Train{train_basin_int}/test_basin/predict"
    output_dir = f'hyper/out/{loc}/LSTM_PUB_random/ensemble/Train{train_basin_int}/test_basin/results' 
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    c = 0
    for seed in seed_list:
        c += 1
        results_eva = []
        for test_basin in test_basins_list:
            input_file = f"{input_dir}/pub_lstm_{test_basin}_val.csv"

            if os.path.exists(input_file):
                df = pd.read_csv(input_file, header=0, index_col=0)
            else:
                print(f"File {input_file} does not exist.")
                continue

            # Check if 'qobs' exists in the index
            if 'qobs' not in df.index:
                print(f"'qobs' is not in the index for test basin {test_basin}. Skipping.")
                continue
            target_eva = df.loc['qobs'].values

            if f'qsim_{seed}' in df.index:
                predict_eva = df.loc[f'qsim_{seed}'].values

                file_results_eva = {'file_num': test_basin}

                for benchmark in benchmark_list:
                    file_results_eva.update({
                        f'{benchmark}': BMK(target_eva, predict_eva, benchmark)
                    })

                results_eva.append(file_results_eva)
            else:
                # Skip if the seed column does not exist
                continue



        results_eva_df = pd.DataFrame(results_eva)
        results_eva_df.set_index("file_num", inplace=True)
        results_eva_df.to_csv(
            f"{output_dir}/LSTM_PUB_results_Train{train_basin_int}_sample{c}_eva.csv", 
            index = True
        )

        all_results.append(results_eva_df)


    # calculate the average of the results for each test basin for each benchmark (an average of 5 seeds)
    all_concat = pd.concat(all_results)
        # Concatenate results across all seeds

    # Compute mean across all seeds for each test basin and benchmark
    results_mean_data = all_concat.groupby(all_concat.index).mean()

    results_mean_data.to_csv(
        f"{output_dir}/LSTM_PUB_results_mean_eva.csv", 
        index = True
    )

    print(f"Results saved to {output_dir} for train_basin_int {train_basin_int} and sample {c}.")
