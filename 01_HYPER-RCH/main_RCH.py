"""
This code is associated to the following paper:

Funato, M., Sawada, Y., "Multi-Model Ensemble and Reservoir Computing for River Discharge Prediction in Ungauged Basins".
currently under submission
"""

# Description: This script is used to run for ESN with input from the ensemble BMA model for the MERV-Jp dataset.
from esn_RCH import ESN
import pandas as pd
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
import os

#### obs flow not included in the input data

#####################
benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]

buf = f""

nexttime = True

input_size = 3
output_size = 1

#loc, ver = "JP", 1
loc, ver = "JP", 2

reservoir_size = 700
spectral_radius = 0.9   
washout = 0
ridge_param = 0.1

file_tag = f"r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}"
#####################
print(file_tag)

ver_name = "ver1_1" if ver == 1 else "ver2_0"
if loc == "JP" and ver == 1:
    file_tot_num  = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87

varssim_dir = f"data/MERVJP/varssim_nocal/ver2_0"

file_list = list(range(1, file_tot_num+1))

output_dir = f'out/{loc}/RCH/{reservoir_size}_{ridge_param}'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/reservoir', exist_ok=True)
os.makedirs(output_dir + '/Wout', exist_ok=True)

if os.path.exists(output_dir + f'/RCH_{file_tag}_log.txt'):
    open(output_dir + f'/RCH_{file_tag}_log.txt', 'w').close()
log_file = open(output_dir + f'/RCH_{file_tag}_log.txt', 'a')

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")

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

#####

BMA_pred_cal = pd.read_csv(f"out/{loc}/BMA/predict/BMA_predict_cal.csv", index_col=0)
BMA_pred_eva = pd.read_csv(f"out/{loc}/BMA/predict/BMA_predict_eva.csv", index_col=0)

results_cal = []
results_eva = []
target_analysis = []

model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.1,
            spectral_radius=spectral_radius,
            input_scale=0.5)

for file_num in file_list:
    print(file_num)
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"file_{file_num}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    ### INPUT DATA ###
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  
    df_eva = df[start_date_eva:end_date_eva].copy()

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal = np.vstack([
        df_cal['Precip'].values,
        df_cal['Temp'].values,
        df_cal['PET'].values
    ])
    input_eva = np.vstack([
        df_eva['Precip'].values,
        df_eva['Temp'].values,
        df_eva['PET'].values
    ])

    
    obs_cal = df_cal['Obs flow'].values.reshape(-1, 1)
    obs_eva = df_eva['Obs flow'].values.reshape(-1, 1)

    #target_cal = np.concatenate((input_cal[3][1:], input_eva[3][:input_cal.shape[1] - 1])).tolist()

    target_cal = np.concatenate([obs_cal[1:], obs_eva[0:1]]) #2922
    target_eva = obs_eva[1:]  #1094

    input_eva = input_eva[:, :-1]

    # Knowledged based model output from cal: 0 ~ n-1, eva: 0 ~ m-1
    knowbsd_cal = BMA_pred_cal.loc[f"file_{file_num}_cal"].values  # (2898,) t
    knowbsd_eva = BMA_pred_eva.loc[f"file_{file_num}_eva"].values  # (1089,) t

    # If knowbsd_cal and knowbsd_eva need to be 2-dimensional
    knowbsd_cal = knowbsd_cal.reshape(-1, 1) # (1, 2898) then transpose to (2898, 1)
    knowbsd_eva = knowbsd_eva.reshape(-1, 1) # (1, 1089) then transpose to (1089, 1) 

    knowbsd_eva = knowbsd_eva[:-1]
    #print("input_cal",input_cal.shape) # 4,2898
    #print("input_eva",input_eva.shape) # 4,1089
    #print("target_cal",target_cal.shape) # 2898,
    #print("target_eva",target_eva.shape) # 1089,
    #print("knwbsd_cal",knowbsd_cal.shape) # 2898,1
    #print("knwbsd_eva",knowbsd_eva.shape) #1090,0

    # CALIBRATION
    # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
    # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW
    # EVALUATION
    # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
    # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

    # input_cal <- 1~n-1, knwbsdsim <- 0~n-1, target_data <-2~n
    W_out, reservoir = model.train(input_cal, knwbsd_sim = knowbsd_cal, target_data=target_cal, washout=washout, ridge_param=ridge_param)
    #WOUT AND RESERVOIR IS UPDATED EVERY FILE CORRECTLY

    reservoir_row = np.concatenate((np.array([f'file_{file_num}']), reservoir.flatten()))
    #reservoir_rows.append(reservoir_row)
    reservoir_df = pd.DataFrame(reservoir_row)
    reservoir_df.to_csv(output_dir + f"/reservoir/RCH_reservoir_{file_tag}_file{file_num}.csv", index=False, header=False)


    # W_out (1,R+1)
    # reservoir (R, )

    W_out_row = np.concatenate(([f'file_{file_num}'], W_out.flatten()))
    #W_out_rows.append(W_out_row)
    W_out_df = pd.DataFrame(W_out_row)
    W_out_df.to_csv(output_dir + f"/Wout/RCH_Wout_{file_tag}_file{file_num}.csv", index=False, header=False)

    # input_eva <- 0~m-1, knwbsd_sim <- 0~m-1
    predict_cal = model.predict(reservoir, input_cal, knwbsd_sim = knowbsd_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
    predict_eva = model.predict(reservoir, input_eva, knwbsd_sim = knowbsd_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

    print(predict_cal)
    print(predict_eva)
    print(BMK(target_cal.T,predict_cal,"KGE"), BMK(target_eva.T, predict_eva,"KGE"))

    file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal.flatten())
    file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva.flatten())

    if file_num == 1:
        predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal[0]))
        predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva[0]))

        date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
        date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

        predict_cal_df = pd.DataFrame([date_row_cal])
        predict_eva_df = pd.DataFrame([date_row_eva])

        predict_cal_df.to_csv(output_dir + f"/predict/RCH_predict_{file_tag}_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/RCH_predict_{file_tag}_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/RCH_predict_{file_tag}_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/RCH_predict_{file_tag}_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'RCH_{file_tag}_{benchmark}_cal': BMK(target_cal.T, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'RCH_{file_tag}_{benchmark}_eva': BMK(target_eva.T, predict_eva, benchmark)
        })

    # Append the completed dictionary to the results list
    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)

    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")

    del W_out, reservoir, predict_eva, predict_cal

log_file.write("DONE\n")
log_file.close()

#reservoir_df = pd.DataFrame(reservoir_rows)
#reservoir_df.to_csv(output_dir + f"/reservoir/RCH_reservoir_{file_tag}.csv", index=False, header=False)

#W_out_df = pd.DataFrame(W_out_rows)
#W_out_df.to_csv(output_dir + f"/Wout/RCH_Wout_{file_tag}.csv", index=False, header=False)
    
results_cal_df = pd.DataFrame(results_cal)
results_cal_df.to_csv(output_dir + f'/results/RCH_results_{file_tag}_cal.csv', index=False)

results_eva_df = pd.DataFrame(results_eva)
results_eva_df.to_csv(output_dir + f'/results/RCH_results_{file_tag}_eva.csv', index=False)

print(f"RCH_{file_tag}")
print("DONE")