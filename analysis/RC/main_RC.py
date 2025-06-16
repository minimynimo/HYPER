# Description: Main script for running the ESN model on multiple files.
import matplotlib
matplotlib.use('Agg')
from esn_RC import ESN
import pandas as pd
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
import os

#####################
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

nexttime = True

buf = ""

input_size = 3
output_size = 1
reservoir_size = 700
#reservoir_size = 200
spectral_radius = 0.4
washout = 0
#ridge_param = 1.0  
ridge_param = 0.001 

#loc, ver = "JP", 1
loc, ver = "JP", 2

file_tag = f"_r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}"
#####################
print(file_tag)

ver_name = "ver1_1" if ver == 1 else "ver2_0"
if loc == "JP" and ver == 1:
    file_tot_num  = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87
varssim_dir = f"hyper/data/MERVJP/varssim_nocal/{ver_name}"

file_list = list(range(1, file_tot_num+1))

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

output_dir = f'hyper/out/{loc}/RC_{reservoir_size}_{ridge_param}'

os.makedirs(output_dir + '/results', exist_ok=True)
os.makedirs(output_dir + '/predict', exist_ok=True)
os.makedirs(output_dir + '/Wout', exist_ok=True)


def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

def BMK(obs_data,sim_data,benchmark):
    """
    obs_data = np.array(obs_data)
    sim_data = np.array(sim_data)
    
    mask = ~np.isnan(obs_data) & ~np.isnan(sim_data)
    obs_data = obs_data[mask]
    sim_data = sim_data[mask]
    """
    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data)
        sim_std = np.std(sim_data)
        if obs_std == 0:
            obs_std = 1e-6
        if obs_ave == 0:
            obs_ave = 1e-6
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

if os.path.exists(output_dir + f'/RC{file_tag}_log.txt'):
    open(output_dir + f'/RC{file_tag}_log.txt', 'w').close()
log_file = open(output_dir + f'/RC{file_tag}_log.txt', 'a')

results_cal = []
results_eva = []

model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)

# Initialize the Date columns for calibration and evaluation predictions


for file_num in file_list:
    print(file_num)
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"file_{file_num}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()


    ### INPUT DATA ###
    df = load_data(file_num)

    df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
    df_eva = df[start_date_eva:end_date_eva].copy()

    # Observations
    #obs_cal = masked_array(df_cal['Obs flow'].values, mask=(df_cal['Obs flow'].values == -999))
    #obs_eva = masked_array(df_eva['Obs flow'].values, mask=(df_eva['Obs flow'].values == -999))
    # Remove NaNs from observations and corresponding simulations
    #obs_cal = obs_cal[~np.isnan(obs_cal)]
    #df_cal = df_cal.dropna(subset=['Obs flow'])
    #obs_eva = obs_eva[~np.isnan(obs_eva)]
    #df_eva = df_eva.dropna(subset(['Obs flow'])

    # For each timestep, Precip,Temp,PET,Obs flow
    input_cal = np.hstack([
        df_cal['Precip'].values.reshape(-1, 1),
        df_cal['Temp'].values.reshape(-1, 1),
        df_cal['PET'].values.reshape(-1, 1)]).T  # 4,2922
    
    input_eva = np.hstack([
        df_eva['Precip'].values.reshape(-1, 1),
        df_eva['Temp'].values.reshape(-1, 1),
        df_eva['PET'].values.reshape(-1, 1)]).T # 3,1095
    
    obs_cal = df_cal['Obs flow'].values.reshape(-1, 1)
    obs_eva = df_eva['Obs flow'].values.reshape(-1, 1)

    #target_cal = np.concatenate((input_cal[3][1:], input_eva[3][:input_cal.shape[1] - 1])).tolist()

    target_cal = np.concatenate([obs_cal[1:], obs_eva[0:1]]) #2922
    target_eva = obs_eva[1:]  #1094

    input_eva = input_eva[:, :-1]

    #print("input_cal",input_cal.shape) # 4,2898
    #print("input_eva",input_eva.shape) # 4,1089
    #print("target_cal",target_cal.shape) # 2898,
    #print("target_eva",target_eva.shape) # 1089,

    ##########!!!!
    #data = readBinary(path, precision=4, nsteps=total_length*time_thinning_step, npoints=output_size*space_thinning_step)
    #data = data.getData(step=time_thinning_step).T[::space_thinning_step, :]

    # CALIBRATION
    # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
    # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW
    # EVALUATION
    # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
    # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

    W_out, reservoir = model.train(input_cal, target_data=target_cal, washout=washout, ridge_param=ridge_param)

    # W_out (1,R)
    # reservoir (R, )

    W_out_row = np.concatenate(([f'file_{file_num}'], W_out.flatten()))
    W_out_df = pd.DataFrame([W_out_row])
    if file_num == 1:
        W_out_df.to_csv(output_dir + f"/Wout/RC_Wout{file_tag}.csv", mode='w', index=False, header=False)
    else:
        W_out_df.to_csv(output_dir + f"/Wout/RC_Wout{file_tag}.csv", mode='a', index=False, header=False)

    predict_cal = model.predict(reservoir, input_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
    predict_eva = model.predict(reservoir, input_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
    
    #print(predict_cal.shape) # 1,2922
    #print(predict_eva.shape) # 1,1094
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

        predict_cal_df.to_csv(output_dir + f"/predict/RC_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
        predict_eva_df.to_csv(output_dir + f"/predict/RC_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

    with open(output_dir + f"/predict/RC_predict{file_tag}_cal.csv", 'a') as file:
        pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
    with open(output_dir + f"/predict/RC_predict{file_tag}_eva.csv", 'a') as file:
        pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

    file_results_cal = {'file_num': file_num}
    file_results_eva = {'file_num': file_num}

    # Loop over each benchmark and add results to the current file's dictionary
    for benchmark in benchmark_list:
        file_results_cal.update({
            f'RC{file_tag}_{benchmark}_cal': BMK(target_cal.T, predict_cal, benchmark)
        })
        file_results_eva.update({
            f'RC{file_tag}_{benchmark}_eva': BMK(target_eva.T, predict_eva, benchmark)
        })

    results_cal.append(file_results_cal)
    results_eva.append(file_results_eva)
    
    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")
    log_file.flush()
    del W_out, reservoir, predict_eva, predict_cal

df_results_cal = pd.DataFrame(results_cal)
df_results_cal.to_csv(output_dir + f'/results/RC_results{file_tag}_cal.csv', index=False, header=True)

df_results_eva = pd.DataFrame(results_eva)
df_results_eva.to_csv(output_dir + f'/results/RC_results{file_tag}_eva.csv', index=False, header=True)


log_file.write("DONE\n")
log_file.close()

print(f"RC{file_tag}")
print("DONE")