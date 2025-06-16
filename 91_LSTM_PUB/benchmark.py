import pandas as pd
import numpy as np
import os

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

#loc, ver = "JP", 1
loc, ver = "JP", 2


output_dir = f'hyper/out/{loc}/LSTM_PUB/results/'

input_dir = f"hyper/out/JP/LSTM_PUB/ensemble"

ver_name = "ver1_1" if ver == 1 else "ver2_0"   

if loc == "JP" and ver == 1:
    file_tot_num  = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87

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

results_eva = []
for file_num in file_num_list:
    input_file = f"{input_dir}/pub_lstm_{file_num}_val_2.csv"
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
    else:
        print(f"File {input_file} does not exist.")
        continue

    target_eva = df['qobs'].values
    predict_eva = df['qsim'].values
    
    file_results_eva = {'file_num': file_num}

    for benchmark in benchmark_list:
        file_results_eva.update({
            f'LSTM_PUB_{benchmark}_eva': BMK(target_eva, predict_eva, benchmark)
        })

    results_eva.append(file_results_eva)

results_eva_df = pd.DataFrame(results_eva)
results_eva_df.to_csv(
    f"{output_dir}/LSTM_PUB_results_eva.csv",
    index=False
)
print(f"Results saved to {output_dir}/LSTM_PUB_results_eva.csv")

