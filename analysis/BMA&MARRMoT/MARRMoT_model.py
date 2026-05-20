# Calculates the performance of each model in the ensemble using the benchmarks NSE, KGE, logNSE, E1, Erel, VE, d, RMSE, and MAE.
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pandas as pd
from numpy.ma import masked_array
import warnings
warnings.filterwarnings("ignore")


############
benchmark_list = ['NSE', 'KGE','E1', 'VE', 'd','RMSE', 'MAE']


#loc, ver = "JP", 1
loc, ver = "JP", 2
############

nocal_tag = "_nocal" ""

ver_name = 'ver1_1' if ver == 1 else 'ver2_0'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87

varssim_dir = f"data/MERVJP/varssim{nocal_tag}/{ver_name}"

# Define the calibration and evaluation periods
start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'


model_list = [f"m{i:02d}" for i in range(1, 48)]

buf = ""

output_dir = f'out/{loc}/MARRMoT{nocal_tag}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Helper function to format file names
def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

# Function to load and preprocess data
def load_data(file_num):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    try:
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    except:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

def BMK(obs_data, sim_data, benchmark):
    obs_data = np.array(obs_data)
    sim_data = np.array(sim_data)

    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]

    if benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data) + 1e-8  # Add small value to avoid division by zero
        sim_std = np.std(sim_data) + 1e-8  # Add small value to avoid division by zero
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2) 
    
    elif benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "VE":
        return 1 - np.sum(np.abs(obs_data - sim_data)) / np.sum(obs_data)
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave))) + 1e-8
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))


# Run BMA for each dataset
for model_name in model_list:
    # no need for predict. only need results
    results_cal = []
    results_eva = []

    for file_num in range(1, file_tot_num+1):
        # Load the data
        df = load_data(file_num)

        df_cal = df[start_date_cal:end_date_cal].copy()  
        df_eva = df[start_date_eva:end_date_eva].copy()

        df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_eva = df[start_date_eva:end_date_eva].copy()

        # Observations
        obs_cal = df_cal['Obs flow'].values
        obs_eva = df_eva['Obs flow'].values

        sim_cal = df_cal[model_name].values
        sim_eva = df_eva[model_name].values
        
        file_results_cal = {'file_num': file_num}
        file_results_eva = {'file_num': file_num}

        # Loop over each benchmark and add results to the current file's dictionary
        for benchmark in benchmark_list:
            file_results_cal.update({
                f'{model_name}_{benchmark}_cal': BMK(obs_cal, sim_cal, benchmark)
            })
            file_results_eva.update({
                f'{model_name}_{benchmark}_eva': BMK(obs_eva, sim_eva, benchmark)
            })  


        # Append the completed dictionary to the results list
        results_cal.append(file_results_cal)
        results_eva.append(file_results_eva)

    df_results_cal = pd.DataFrame(results_cal)
    df_results_cal.to_csv(output_dir + f'/{model_name}_results_cal.csv', index=False)

    df_results_eva = pd.DataFrame(results_eva)
    df_results_eva.to_csv(output_dir + f'/{model_name}_results_eva.csv', index=False)

print("DONE")