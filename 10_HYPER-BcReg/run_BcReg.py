# BC-PCA model 
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def run_BC_pre_PCA(model, file_list, loc, output_dir, bma_cal_og, bma_eva_og, start_date_cal, end_date_cal, start_date_eva, end_date_eva, benchmark_list, file_tag, washout, ridge_param, spin, nexttime, varssim_dir, log_file=None):
    result_cal_og = []
    result_eva_og = []
    W_out_og_rows = []

    for file_num in file_list:
        print(f'file {file_num}')

        if log_file:
            start_time = datetime.now()
            start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
            log_file.write(f"BC-PCA og FILE_{file_num}\n")
            log_file.write(f"start: {start_time_st}\n")
            log_file.flush()

        ### INPUT DATA ###
        df_og = load_data(file_num, varssim_dir, loc, start_date_cal, end_date_eva)

        df_cal_og = df_og[start_date_cal:end_date_cal].copy()  # Use copy to avoid SettingWithCopyWarning
        df_eva_og = df_og[start_date_eva:end_date_eva].copy()

        df_cal_og['Sim flow'] = bma_cal_og[f'file_{file_num}_cal'].values
        df_eva_og['Sim flow'] = bma_eva_og[f'file_{file_num}_eva'].values

        # Calculate Flow Error
        df_cal_og['Flow Error'] = df_cal_og['Obs flow'] - df_cal_og['Sim flow']
        df_eva_og['Flow Error'] = df_eva_og['Obs flow'] - df_eva_og['Sim flow']

        # For each timestep, Precip,Temp,PET,Obs flow
        input_cal_og = np.hstack([
            df_cal_og['Precip'].values.reshape(-1, 1),
            df_cal_og['Temp'].values.reshape(-1, 1),
            df_cal_og['PET'].values.reshape(-1, 1)]).T # Observed - Simulation
        input_eva_og = np.hstack([
            df_eva_og['Precip'].values.reshape(-1, 1),
            df_eva_og['Temp'].values.reshape(-1, 1),
            df_eva_og['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

        obs_error_cal_og = df_cal_og['Flow Error'].values.reshape(-1, 1)
        obs_error_eva_og = df_eva_og['Flow Error'].values.reshape(-1, 1)

        # target error
        error_target_cal_og = np.concatenate([obs_error_cal_og[1:], obs_error_eva_og[0:1]]) #2898
        error_target_eva = obs_error_eva_og[1:]  #1089

        # Target flow
        target_cal_og = np.concatenate((df_cal_og['Obs flow'].values[1:], df_eva_og['Obs flow'].values[0:1])) #2898
        target_eva_og = np.concatenate((df_eva_og['Obs flow'].values[1:],)) #1089

        input_eva_og = input_eva_og[:, :-1]

        #############################
        # CALIBRATION
        W_out_og, reservoir_og = model.train(input_cal_og, target_data=error_target_cal_og, washout=washout, ridge_param=ridge_param, spinoff=spin)
        W_out_og_row = np.concatenate(([f'file_{file_num}'], W_out_og.flatten()))
        W_out_og_rows.append(W_out_og_row)
        
        ### PREDICTIONS OF ERRORS ###
        error_cal_og = model.predict(reservoir_og, input_cal_og, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        error_eva_og = model.predict(reservoir_og, input_eva_og, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        predict_cal_og = error_cal_og + np.concatenate([df_cal_og['Sim flow'].values[1:], df_eva_og['Sim flow'].values[0:1]])
        predict_eva_og = error_eva_og + df_eva_og['Sim flow'].values[1:]

        predict_cal_og[predict_cal_og < 0] = 0
        predict_eva_og[predict_eva_og < 0] = 0

        # Ensure no division by zero
        predict_cal_og = np.where(predict_cal_og == 0, 1e-6, predict_cal_og)
        predict_eva_og = np.where(predict_eva_og == 0, 1e-6, predict_eva_og)

        #print(predict_cal_og)
        #print(predict_eva_og)
        #print(BMK(target_cal_og.T, predict_cal_og, "KGE"), BMK(target_eva_og.T, predict_eva_og, "KGE"))
        """
        file_row_cal = [f'file_{file_num}_cal'] + list(predict_cal_og.flatten())
        file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva_og.flatten())

        if file_num == file_list[0]:
            predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(predict_cal_og[0]))
            predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva_og[0]))

            date_row_cal = list(['Date'] + [str(date.date()) for date in predict_cal_dates])
            date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])

            predict_cal_df = pd.DataFrame([date_row_cal])
            predict_eva_df = pd.DataFrame([date_row_eva])

            predict_cal_df.to_csv(output_dir + f"/train_basin/predict/BC-PCA-lasso_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
            predict_eva_df.to_csv(output_dir + f"/train_basin/predict/BC-PCA-lasso_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/train_basin/predict/BC-PCA-lasso_predict{file_tag}_cal.csv", 'a') as file:
            pd.DataFrame([file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/train_basin/predict/BC-PCA-lasso_predict{file_tag}_eva.csv", 'a') as file:
            pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)"""

        file_result_cal_og = {'file_num': file_num}
        file_result_eva_og = {'file_num': file_num}

        # Ensure target_cal_og and predict_cal_og are single vectors with the same length
        predict_cal_og = predict_cal_og.flatten()
        predict_eva_og = predict_eva_og.flatten()

        for benchmark in benchmark_list:
            file_result_cal_og[f'BC_og{file_tag}_{benchmark}_cal'] = BMK(target_cal_og, predict_cal_og, benchmark)
            file_result_eva_og[f'BC_og{file_tag}_{benchmark}_eva'] = BMK(target_eva_og, predict_eva_og, benchmark)

        result_cal_og.append(file_result_cal_og)
        result_eva_og.append(file_result_eva_og)

        if log_file:
            end_time = datetime.now()
            end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
            log_file.write(f"end: {end_time_st}\n")
            log_file.write(f"elapsed: {end_time - start_time}\n")

        del W_out_og, reservoir_og, predict_eva_og, predict_cal_og

    return result_cal_og, result_eva_og, W_out_og_rows

def PCA_lasso(pc_values, attribute_values, file_list_train, file_list_test, n_components, alpha, fs, output_dir, output_fig_dir, columns_drop, sample = False):
    file_list = attribute_values['File_num']
    # exclude 'File_num' form columns_drop
    attribute_values = attribute_values.drop(columns=columns_drop)  # Drop non-numeric columns
    attribute_values = attribute_values.apply(pd.to_numeric, errors='coerce')

    #normalize
    scaler = StandardScaler()
    attribute_values_scaled = scaler.fit_transform(attribute_values)
    attribute_values_scaled = pd.DataFrame(attribute_values_scaled, columns= attribute_values.columns)

    attribute_values_scaled.insert(0, 'File_num', file_list.values)

    attribute_values_train = attribute_values_scaled[attribute_values_scaled['File_num'].isin(file_list_train)]
    attribute_values_test = attribute_values_scaled[attribute_values_scaled['File_num'].isin(file_list_test)]

    # drop non-numeric columns
    attribute_values_train = attribute_values_train.drop(columns='File_num')
    attribute_values_test = attribute_values_test.drop(columns='File_num') 

    # Handle non-numeric values
    attribute_values_train = attribute_values_train.apply(pd.to_numeric, errors='coerce')
    attribute_values_train = attribute_values_train.fillna(attribute_values_train.mean())

    attribute_values_test = attribute_values_test.apply(pd.to_numeric, errors='coerce')
    attribute_values_test = attribute_values_test.fillna(attribute_values_test.mean())

    # Ensure the number of components is less than or equal to the number of attributes

    lasso_models = {} # Store the Lasso models for each PC based on the train data
    predicted_pcs_train = {} # Store the predicted PC values for each PC based on the train data
    predicted_pcs_test = {} # Store the predicted PC values for each PC based on the test data
    for pc_n in range(1, n_components+1):  # Skip the 'file_num' column
        y = pc_values[f'PC{pc_n}']

        lasso = Lasso(alpha=alpha, max_iter=200000)  # Increase max_iter to address convergence warning
        lasso.fit(attribute_values_train, y)
        lasso_models[f'PC{pc_n}'] = lasso # Store the Lasso model for each PC

        # Get non-zero coefficients and their corresponding attribute names
        non_zero_coefs = lasso.coef_[lasso.coef_ != 0]
        non_zero_attributes = attribute_values_train.columns[lasso.coef_ != 0]

        # Sort coefficients and attributes by absolute value of coefficients
        sorted_indices = np.argsort(np.abs(non_zero_coefs))[::-1]
        sorted_coefs = non_zero_coefs[sorted_indices]
        sorted_attributes = non_zero_attributes[sorted_indices]

        # save to csv
        # lasso_coefs = pd.DataFrame({'Attributes': sorted_attributes, 'Coefficients': sorted_coefs})
        # lasso_coefs.to_csv(f'{output_dir}/lasso_coefficients_PC{pc_n}.csv', index=False)

        # Plot the coefficients
        plt.figure(figsize=(20, 6))
        plt.bar(sorted_attributes, sorted_coefs, color = 'royalblue')
        plt.xlabel('Attributes', fontsize=fs)
        plt.ylabel('Lasso Coefficients', fontsize=fs)
        plt.title(f'Lasso Coefficients for PC{pc_n}', fontsize=fs)
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_fig_dir}/lasso_coefficients_train_PC{pc_n}.png')
        plt.close()

        # Plot the actual vs predicted PC values 
        predicted_pcs_train[f'PC{pc_n}'] = lasso_models[f'PC{pc_n}'].predict(attribute_values_train)
        predicted_pcs_test[f'PC{pc_n}'] = lasso_models[f'PC{pc_n}'].predict(attribute_values_test)

        plt.figure(figsize=(7, 7))
        plt.scatter(y, predicted_pcs_train[f'PC{pc_n}'], color='royalblue')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='gray', linestyle='--', label='y=x')
        plt.xlabel(f'Original PC{pc_n} values', fontsize=fs)
        plt.ylabel(f'Predicted PC{pc_n} values', fontsize=fs)
        plt.title(f'Training Basin: Original PC{pc_n} vs Predicted PC{pc_n}', fontsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()

        plt.savefig(f'{output_fig_dir}/original_vs_predicted_train_PC{pc_n}.png')

        plt.close()

    # save lasso model to csv
    lasso_models_df = pd.DataFrame({pc: model.coef_ for pc, model in lasso_models.items()})
    lasso_models_df.index = attribute_values_train.columns

    if not os.path.exists(f'{output_dir}/lasso_coefficient'):
        os.makedirs(f'{output_dir}/lasso_coefficient')
    if sample:
        lasso_models_df.to_csv(f'{output_dir}/lasso_coefficient/lasso_coefficient_sample{sample}.csv', index=True)
    else:
        lasso_models_df.to_csv(f'{output_dir}/lasso_coefficient/lasso_coefficient.csv', index=True)

    return lasso_models, predicted_pcs_train, predicted_pcs_test

def PCA_lasso_two_rgn(pc_values, attribute_values_train,attribute_values_test, n_components, alpha, fs, output_dir, output_fig_dir,  sample = False):
    # Handle non-numeric values
    attribute_values_train = attribute_values_train.apply(pd.to_numeric, errors='coerce')
    attribute_values_train = attribute_values_train.fillna(attribute_values_train.mean())

    attribute_values_test = attribute_values_test.apply(pd.to_numeric, errors='coerce')
    attribute_values_test = attribute_values_test.fillna(attribute_values_test.mean())

    lasso_models = {} # Store the Lasso models for each PC based on the train data
    predicted_pcs_train = {} # Store the predicted PC values for each PC based on the train data
    predicted_pcs_test = {} # Store the predicted PC values for each PC based on the test data
    for pc_n in range(1, n_components+1):  # Skip the 'file_num' column
        y = pc_values[f'PC{pc_n}']

        lasso = Lasso(alpha=alpha, max_iter=200000)  # Increase max_iter to address convergence warning
        lasso.fit(attribute_values_train, y)
        lasso_models[f'PC{pc_n}'] = lasso # Store the Lasso model for each PC

        # Get non-zero coefficients and their corresponding attribute names
        non_zero_coefs = lasso.coef_[lasso.coef_ != 0]
        non_zero_attributes = attribute_values_train.columns[lasso.coef_ != 0]

        # Sort coefficients and attributes by absolute value of coefficients
        sorted_indices = np.argsort(np.abs(non_zero_coefs))[::-1]
        sorted_coefs = non_zero_coefs[sorted_indices]
        sorted_attributes = non_zero_attributes[sorted_indices]

        # save to csv
        # lasso_coefs = pd.DataFrame({'Attributes': sorted_attributes, 'Coefficients': sorted_coefs})
        # lasso_coefs.to_csv(f'{output_dir}/lasso_coefficients_PC{pc_n}.csv', index=False)

        # Plot the coefficients
        plt.figure(figsize=(20, 6))
        plt.bar(sorted_attributes, sorted_coefs, color = 'royalblue')
        plt.xlabel('Attributes', fontsize=fs)
        plt.ylabel('Lasso Coefficients', fontsize=fs)
        plt.title(f'Lasso Coefficients for PC{pc_n}', fontsize=fs)
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_fig_dir}/lasso_coefficients_train_PC{pc_n}.png')
        plt.close()

        # Plot the actual vs predicted PC values 
        predicted_pcs_train[f'PC{pc_n}'] = lasso_models[f'PC{pc_n}'].predict(attribute_values_train)
        predicted_pcs_test[f'PC{pc_n}'] = lasso_models[f'PC{pc_n}'].predict(attribute_values_test)

        plt.figure(figsize=(7, 7))
        plt.scatter(y, predicted_pcs_train[f'PC{pc_n}'], color='royalblue')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='gray', linestyle='--', label='y=x')
        plt.xlabel(f'Original PC{pc_n} values', fontsize=fs)
        plt.ylabel(f'Predicted PC{pc_n} values', fontsize=fs)
        plt.title(f'Training Basin: Original PC{pc_n} vs Predicted PC{pc_n}', fontsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()

        plt.savefig(f'{output_fig_dir}/original_vs_predicted_train_PC{pc_n}.png')

        plt.close()

    # save lasso model to csv
    lasso_models_df = pd.DataFrame({pc: model.coef_ for pc, model in lasso_models.items()})
    lasso_models_df.index = attribute_values_train.columns

    if not os.path.exists(f'{output_dir}/lasso_coefficient'):
        os.makedirs(f'{output_dir}/lasso_coefficient')
    if sample:
        lasso_models_df.to_csv(f'{output_dir}/lasso_coefficient/lasso_coefficient_sample{sample}.csv', index=True)
    else:
        lasso_models_df.to_csv(f'{output_dir}/lasso_coefficient/lasso_coefficient.csv', index=True)

    return lasso_models, predicted_pcs_train, predicted_pcs_test

def run_BC_post_PCA(model, file_list, loc, output_dir, train_test, PC_n, weights_inverted, model_list, start_date_eva, end_date_eva, benchmark_list, nexttime, file_tag, varssim_dir, log_file = None, sample = False):   
    result_eva_rev = []
    reservoir_rev_rows = []
    
    for file_num in file_list: # 136
        print(f'    file{file_num}')
        if log_file:
            start_time = datetime.now()
            start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
            log_file.write(f"   PC_{PC_n} FILE_{file_num}\n")
            log_file.write(f"   start: {start_time_st}\n")
            log_file.flush()

        BMA_weights_rev = weights_inverted.loc[f'file_{file_num}'].values[:len(model_list)] ### NOTE THAT THE SUM OF THE WEIGHTS WILL NOT NECCESSARILY BE 1 IN THIS CASE
        # scale the BMA_weights to sum to 1
        BMA_weights_rev = BMA_weights_rev / np.sum(BMA_weights_rev)
        RC_weights_rev = weights_inverted.loc[f'file_{file_num}'].values[len(model_list):]
        
        ### INPUT DATA ### file_num, varssim_dir, loc, start_date, end_date
        df_rev = load_data(file_num, varssim_dir, loc, start_date_eva, end_date_eva)
        df_eva_rev = df_rev[start_date_eva:end_date_eva].copy()

        # loading the sim flow data for the BMA and 44 MARRMoT models
        # Load BMA predictions
        bma_df_eva_rev = BMA_revert(df_eva_rev, BMA_weights_rev, model_list)
        df_eva_rev['Sim flow'] = bma_df_eva_rev

        # Calculate Flow Error
        df_eva_rev['Flow Error'] = df_eva_rev['Obs flow'] - df_eva_rev['Sim flow']

        # For each timestep, Precip,Temp,PET,Obs flow
        input_eva_rev = np.hstack([
            df_eva_rev['Precip'].values.reshape(-1, 1),
            df_eva_rev['Temp'].values.reshape(-1, 1),
            df_eva_rev['PET'].values.reshape(-1, 1)]).T # Observed - Simulation

        obs_error_eva_rev = df_eva_rev['Flow Error'].values.reshape(-1, 1)

        # Target flow
        target_eva_rev = np.concatenate((df_eva_rev['Obs flow'].values[1:],)) #1089

        input_eva_rev = input_eva_rev[:, :-1]

        ##########
        # EVALUATION
        # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        # input_eva <- 0~m-1, knwbsd_sim <- 0~m-1
        ### PREDICTIONS OF ERRORS ###
        error_eva_rev, reservoir_last = model.predict_PCA(RC_weights_rev, input_eva_rev, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        reservoir_rev_rows.append(np.concatenate(([f'file_{file_num}'], reservoir_last.flatten())))

        # もとの入力値のtimestep = t のとき、error_cal&evaのtimestep = t+1 になるようにしている
        # predict = error_cal + BMAsim
        # predict のtimestep = t+1
        predict_eva_rev = error_eva_rev + df_eva_rev['Sim flow'].values[1:]

        predict_eva_rev[predict_eva_rev < 0] = 0

        # Ensure no division by zero
        predict_eva_rev = np.where(predict_eva_rev == 0, 1e-6, predict_eva_rev)

        #print(predict_eva_rev)
        #print(BMK(target_eva_rev.T, predict_eva_rev,"KGE"))

        file_row_eva = [f'file_{file_num}_eva'] + list(predict_eva_rev.flatten())

        if file_num == file_list[0]:
            predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_eva_rev[0]))
            date_row_eva = list(['Date'] + [str(date.date()) for date in predict_eva_dates])
            predict_eva_df = pd.DataFrame([date_row_eva])
            if sample:
                predict_eva_df.to_csv(output_dir + f"/{train_test}_basin/predict/sample{sample}/BC-PCA-lasso_predict{file_tag}_rev_PC{PC_n}_eva.csv", mode='w', index=False, header=False)
            else:
                predict_eva_df.to_csv(output_dir + f"/{train_test}_basin/predict/BC-PCA-lasso_predict{file_tag}_rev_PC{PC_n}_eva.csv", mode='w', index=False, header=False)

        if sample:
            with open(output_dir + f"/{train_test}_basin/predict/sample{sample}/BC-PCA-lasso_predict{file_tag}_rev_PC{PC_n}_eva.csv", 'a') as file:
                pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)
        else:
            with open(output_dir + f"/{train_test}_basin/predict/BC-PCA-lasso_predict{file_tag}_rev_PC{PC_n}_eva.csv", 'a') as file:
                pd.DataFrame([file_row_eva]).to_csv(file, header=False, index=False)

        file_results_eva_rev = {'file_num': file_num}

        predict_eva_rev = predict_eva_rev.flatten()
        # Loop over each benchmark and add results to the current file's dictionary
        for benchmark in benchmark_list:
            file_results_eva_rev.update({
                f'BC-PCA-lasso{file_tag}_{benchmark}_eva': BMK(target_eva_rev, predict_eva_rev, benchmark)
            })  

        # Append the completed dictionary to the results list
        result_eva_rev.append(file_results_eva_rev)

        if log_file:
            end_time = datetime.now()
            end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
            log_file.write(f"   end: {end_time_st}\n")
            log_file.write(f"   elapsed: {end_time - start_time}\n")
            log_file.flush()

        del predict_eva_rev

    return result_eva_rev, reservoir_rev_rows

def variable_names_list(model_list, reservoir_size):
    variable_names = [f'BMA_{model}' for model in model_list] + [f'RC_{i}' for i in range(1, reservoir_size+1)]
    variable_names = np.array(variable_names)
    return variable_names

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num, varssim_dir, loc, start_date, end_date):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date, end=end_date)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

def BMK(obs_data, sim_data, benchmark):    
    if benchmark == "NSE":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.square(obs_data - sim_data)) / np.sum(np.square(obs_data - obs_ave)))
    
    elif benchmark == "KGE":
        r = np.corrcoef(obs_data, sim_data)[0, 1]
        obs_ave = np.mean(obs_data)
        sim_ave = np.mean(sim_data)
        obs_std = np.std(obs_data, axis=0)  # Specify axis=0
        sim_std = np.std(sim_data, axis=0)  # Specify axis=0
        obs_std = 1e-6 if obs_std == 0 else obs_std
        obs_ave = 1e-6 if obs_ave == 0 else obs_ave
        return 1 - np.sqrt((r - 1)**2 + ((sim_std / obs_std) - 1)**2 + ((sim_ave / obs_ave) - 1)**2)
    
    elif benchmark == "E1":
        obs_ave = np.mean(obs_data)
        return 1 - (np.sum(np.abs(obs_data - sim_data)) / np.sum(np.abs(obs_data - obs_ave)))
    
    elif benchmark == "VE":
        return 1 - np.sum(np.abs(obs_data - sim_data), axis=0) / np.sum(obs_data, axis=0)  # Specify axis=0
    
    elif benchmark == "d":
        obs_ave = np.mean(obs_data)
        numer = np.sum(np.square(obs_data - sim_data))
        denom = np.sum(np.square(np.abs(sim_data - obs_ave) + np.abs(obs_data - obs_ave)))
        if denom == 0:
            return float('nan')
        return 1 - numer / denom
    
    elif benchmark == "RMSE":
        return np.sqrt(np.mean(np.square(obs_data - sim_data)))
    
    elif benchmark == "MAE":
        return np.mean(np.abs(obs_data - sim_data))

def weight_vector(bma_weights_df, RC_weights_df, reservoir_size, model_list, file_list):
    variable_names = variable_names_list(model_list, reservoir_size)

    weight_vectors =[]
    for file_num in file_list:
        bma_weights = bma_weights_df[f'file_{file_num}'].values
        #print("bma_weights", bma_weights.shape)
        rc_reservoir_values = RC_weights_df.loc[f'file_{file_num}'].values
        #print("rc_reservoir_values", rc_reservoir_values.shape)
        vector = np.concatenate((bma_weights, rc_reservoir_values))
        weight_vectors.append(vector)

    weight_vectors = pd.DataFrame(weight_vectors) ## 135 basins as rows, 744 variables as columns
    weight_vectors.columns = variable_names
    return weight_vectors

def BayesianModelAveraging(model_list, file_list, loc, start_date_cal, end_date_cal, start_date_eva, end_date_eva, varssim_dir):
    # calculate ensemble output using predict_cal, predict_eva, target_cal, target_eva
    log_likelihoods_cal = np.zeros(len(model_list))
    priors = np.ones(len(model_list)) / len(model_list)  # Uniform priors

    weight = {}
    predict_cal_all = {}
    predict_eva_all = {}

    weight["model"] = model_list
    for file_num in file_list:
        # Load the data
        df = load_data(file_num, varssim_dir, loc, start_date_cal, end_date_eva)

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
