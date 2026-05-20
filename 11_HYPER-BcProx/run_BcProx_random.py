"""
This code is associated to the following paper:

Funato, M., Sawada, Y., "Multi-Model Ensemble and Reservoir Computing for River Discharge Prediction in Ungauged Basins".
currently under submission
"""

import pandas as pd
import numpy as np

def run_BC(model, file_list, bma_df_cal_og, bma_df_eva_og, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc, output_dir, washout, ridge_param, nexttime, benchmark_list):
    results_cal = []
    results_eva = []
    W_out_rows = []

    for file_num in file_list:
        ### INPUT DATA ###
        df_train = load_data(file_num, varssim_dir, start_date_cal, end_date_eva, loc)

        df_train_cal = df_train[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_train_eva = df_train[start_date_eva:end_date_eva].copy()

        df_train_cal['Sim flow'] = bma_df_cal_og[f'file_{file_num}_cal'].values
        df_train_eva['Sim flow'] = bma_df_eva_og[f'file_{file_num}_eva'].values

        # Calculate Flow Error
        df_train_cal['Flow Error'] = df_train_cal['Obs flow'] - df_train_cal['Sim flow']
        df_train_eva['Flow Error'] = df_train_eva['Obs flow'] - df_train_eva['Sim flow']

        # For each timestep, Precip,Temp,PET,Obs flow
        input_train_cal = np.hstack([
            df_train_cal['Precip'].values.reshape(-1, 1),
            df_train_cal['Temp'].values.reshape(-1, 1),
            df_train_cal['PET'].values.reshape(-1, 1)]).T # Observed - Simulation
        input_train_eva = np.hstack([
            df_train_eva['Precip'].values.reshape(-1, 1),
            df_train_eva['Temp'].values.reshape(-1, 1),
            df_train_eva['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

        obs_train_error_cal = df_train_cal['Flow Error'].values.reshape(-1, 1)
        obs_train_error_eva = df_train_eva['Flow Error'].values.reshape(-1, 1)

        # target error
        error_train_target_cal = np.concatenate([obs_train_error_cal[1:], obs_train_error_eva[0:1]]) #2898
        error_train_target_eva = obs_train_error_eva[1:]  #1089

        # Target flow
        target_train_cal = np.concatenate((df_train_cal['Obs flow'].values[1:], df_train_eva['Obs flow'].values[0:1])) #2898
        target_train_eva = np.concatenate((df_train_eva['Obs flow'].values[1:],)) #1089

        input_train_eva = input_train_eva[:, :-1]

        ##########!!!!
        # CALIBRATION
        # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        # EVALUATION
        # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        W_out, reservoir = model.train(input_train_cal, target_data=error_train_target_cal, washout=washout, ridge_param=ridge_param)

        W_out_row = np.concatenate(([f'file_{file_num}'], W_out.flatten()))
        W_out_rows.append(W_out_row)

        error_train_cal = model.predict(reservoir, input_train_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        error_train_eva = model.predict(reservoir, input_train_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        train_predict_cal = error_train_cal + np.concatenate([df_train_cal['Sim flow'].values[1:], df_train_eva['Sim flow'].values[0:1]])
        train_predict_eva = error_train_eva + df_train_eva['Sim flow'].values[1:]

        train_predict_cal[train_predict_cal < 0] = 0
        train_predict_eva[train_predict_eva < 0] = 0    

        """
        train_file_row_cal = [f'file_{file_num}_cal'] + list(train_predict_cal.flatten())
        train_file_row_eva = [f'file_{file_num}_eva'] + list(train_predict_eva.flatten())

        if file_num == file_list[0]:
            train_predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(train_predict_cal[0]))
            train_predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(train_predict_eva[0]))

            train_date_row_cal = list(['Date'] + [str(date.date()) for date in train_predict_cal_dates])
            train_date_row_eva = list(['Date'] + [str(date.date()) for date in train_predict_eva_dates])

            train_predict_cal_df = pd.DataFrame([train_date_row_cal])
            train_predict_eva_df = pd.DataFrame([train_date_row_eva])

            train_predict_cal_df.to_csv(output_dir + f"/train_basin/predict/BC_PUB_predict_cal.csv", mode='w', index=False, header=False)
            train_predict_eva_df.to_csv(output_dir + f"/train_basin/predict/BC_PUB_predict_eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/train_basin/predict/BC_PUB_predict_cal.csv", 'a') as file:
            pd.DataFrame([train_file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/train_basin/predict/BC_PUB_predict_eva.csv", 'a') as file:
            pd.DataFrame([train_file_row_eva]).to_csv(file, header=False, index=False)
        """

        file_results_train_cal = {'file_num': file_num}
        file_results_train_eva = {'file_num': file_num}

        for benchmark in benchmark_list:
            file_results_train_cal.update({
                f'BC_PUB_{benchmark}_cal': BMK(target_train_cal, train_predict_cal, benchmark)
            })
            file_results_train_eva.update({
                f'BC_PUB_{benchmark}_eva': BMK(target_train_eva, train_predict_eva, benchmark)
            })

        results_cal.append(file_results_train_cal)
        results_eva.append(file_results_train_eva)

        del W_out, train_predict_cal, train_predict_eva, error_train_cal, error_train_eva

    return results_cal, results_eva, W_out_rows


def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num, varssim_dir, start_date_cal, end_date_eva, loc):
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

def BayesianModelAveraging(file_list, model_list, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc):
    # calculate ensemble output using predict_cal, predict_eva, target_cal, target_eva
    log_likelihoods_cal = np.zeros(len(model_list))
    priors = np.ones(len(model_list)) / len(model_list)  # Uniform priors

    weight = {}
    predict_cal_all = {}
    predict_eva_all = {}

    weight["model"] = model_list
    for file_num in file_list:
        # Load the data
        df = load_data(file_num, varssim_dir, start_date_cal, end_date_eva, loc)

        df_cal = df[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_eva = df[start_date_eva:end_date_eva].copy()

        # Observations
        obs_cal = df_cal['Obs flow'].values
        obs_eva = df_eva['Obs flow'].values

        # CALIBRATION
        for i, model in enumerate(model_list):
            model_cal = df_cal[model].values
            log_likelihoods_cal[i] = np.sum(log_likelihood(obs_cal, model_cal))

        # Update priors based on log-likelihoods
        max_log_likelihood_cal = np.max(log_likelihoods_cal)  # For numerical stability
        weights_cal = np.exp(log_likelihoods_cal - max_log_likelihood_cal)
        posterior_cal = weights_cal * priors
        posterior_cal /= np.sum(posterior_cal) ##turn the weight sum into 1

        weight[ f'file_{file_num}'] = posterior_cal

        # EVALUATION
        predict_eva = BMA_revert(df_eva, posterior_cal, model_list)
        predict_cal = BMA_revert(df_cal, posterior_cal, model_list)

        predict_cal_all[f'file_{file_num}_cal'] = predict_cal.flatten()
        predict_eva_all[f'file_{file_num}_eva'] = predict_eva.flatten()

    return weight, predict_cal_all, predict_eva_all

def BMA_revert(model_sim, BMA_weights, model_list):
    ## model_sim includes the sim flow data for the BMA and 44 MARRMoT models with the same date index
    bma_df_eva = np.zeros(len(model_sim))  
    for date in model_sim.index:
        BMA_day_prediction = 0
        for i, model_num in enumerate(model_list):
            model_eva = model_sim.at[date, model_num]
            BMA_day_prediction += BMA_weights[i] * model_eva
        bma_df_eva[model_sim.index.get_loc(date)] = BMA_day_prediction
    return bma_df_eva

def log_likelihood(obs, sim):
    obs = np.where(obs == 0, 1e-6, obs)
    sim = np.where(sim == 0, 1e-6, sim)
    residuals = obs - sim
    sigma2 = np.var(residuals)
    if sigma2 == 0:  # Avoid division by zero
        sigma2 = 10e-6
    return -0.5 * (residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))

def split(a_list, n):
    # devide a_list into roughly n equal parts
    k, m = divmod(len(a_list), n)
    return list(a_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
