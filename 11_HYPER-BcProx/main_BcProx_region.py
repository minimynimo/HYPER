# Description: Main script for running the ESN model on multiple files.
import matplotlib
matplotlib.use('Agg')
from esn_BcProx import ESN
import pandas as pd
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
import os
from run_BcProx import BMK, load_data, BayesianModelAveraging

#####################
benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]

#loc, ver = "JP", 1
loc, ver = "JP", 2

nexttime = True

BMA_data_exist = True

input_size = 3
output_size = 1

spectral_radius = 0.4
washout = 0
ridge_param = 1.0

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'
#####################
ver_name = "ver1_1" if ver == 1 else "ver2_0" 
attribute_dir = f'data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
    region_list = ['Hokkaido', 'Tohoku', 'North Central', 'South Central', 'West']
    region_list_df = pd.read_csv(f'data/river_basin/dataset_{loc}/pub_region_list_{ver_name}.csv')
    distance_matrix = pd.read_csv('data/distance_matrix_v2_sorted.csv', index_col=0, header=0)
    
varssim_dir = f"data/MERVJP/varssim_nocal/{ver_name}"
file_tag = f"_r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}"
print(file_tag)

varssim_dir = f"data/MERVJP/varssim_nocal/{ver_name}"

file_list = list(range(1, file_tot_num+1))

model_list = [f"m{i:02d}" for i in range(1, 48)]

if BMA_data_exist:
    bma_weights_df = pd.read_csv(f"out/{loc}/BMA/weights/BMA_weights.csv", index_col=0, header=0)
    bma_predict_cal_df = pd.read_csv(f"out/{loc}/BMA/predict/BMA_predict_cal.csv", index_col=0, header=0)
    bma_predict_eva_df = pd.read_csv(f"out/{loc}/BMA/predict/BMA_predict_eva.csv", index_col=0, header=0)

    bma_weights_df = bma_weights_df.to_dict(orient='index')
    bma_predict_cal_df = bma_predict_cal_df.to_dict(orient='index')
    bma_predict_eva_df = bma_predict_eva_df.to_dict(orient='index')


model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.1,
            spectral_radius=spectral_radius,
            input_scale=0.5)

output_base_dir = f'out/{loc}/BcProx/region/{reservoir_size}_{ridge_param}'
os.makedirs(output_base_dir, exist_ok=True)
param_file_path = os.path.join(output_base_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")


for region in region_list:
    output_dir = f'{output_base_dir}/{region}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir in ['train_basin/results', 'train_basin/predict',  'train_basin/Wout' ,'test_basin/results', 'test_basin/predict']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    file_tag = f"_r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}"
    print(file_tag)

    results_train_cal = []
    results_train_eva = []
    results_test_eva = []
    W_out_rows = []

    # train_basins is the list of basins to be trained in the certain start and step value
    train_basins = sorted(region_list_df[region_list_df['Region'] == region]['File_num'].tolist())
    train_basins = [file_num for file_num in train_basins if file_num not in test_basins_list]
    print(f'region: {region}, {len(train_basins)} basins')
    print(train_basins)
    test_basins = [num for num in file_list if num not in train_basins] # removed basins in region and test_basins_list

    train_basins_str = list(map(str, train_basins))

    closest_train_basins_df = pd.DataFrame(columns=['TestBasin', 'ClosestTrainBasin'])
    for test_basin in test_basins_list:
        closest_train_basin = distance_matrix.loc[test_basin, train_basins_str].idxmin()
        new_row = pd.DataFrame([{'TestBasin': test_basin, 'ClosestTrainBasin': closest_train_basin}])
        closest_train_basins_df = pd.concat([closest_train_basins_df, new_row], ignore_index=True)

    if BMA_data_exist:
        bma_df_cal_og = {f'file_{train_basin}_cal': np.array(list(bma_predict_cal_df[f'file_{train_basin}_cal'].values())) for train_basin in train_basins}
        bma_df_eva_og = {f'file_{train_basin}_eva': np.array(list(bma_predict_eva_df[f'file_{train_basin}_eva'].values())) for train_basin in train_basins}
        bma_weights_og = {f'file_{train_basin}': bma_weights_df[f'file_{train_basin}'] for train_basin in train_basins}
    else:
        bma_weights_og, bma_df_cal_og, bma_df_eva_og = BayesianModelAveraging(model_list, train_basins, loc, start_date_cal, end_date_cal, start_date_eva, end_date_eva, varssim_dir)

    bma_df_cal_og = pd.DataFrame(bma_df_cal_og)
    bma_df_eva_og = pd.DataFrame(bma_df_eva_og)
    bma_weights_og = pd.DataFrame(bma_weights_og)

    if os.path.exists(output_dir + f'/BcProx{file_tag}_log.txt'):
        open(output_dir + f'/BcProx{file_tag}_log.txt', 'w').close()
    log_file = open(output_dir + f'/BcProx{file_tag}_log.txt', 'a')

    for file_num in train_basins: # go through only the training basins
        print("training: ", file_num)
        start_time = datetime.now()
        start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"FILE_{file_num}\n")
        log_file.write(f"start: {start_time_st}\n")
        log_file.flush()

        test_basins = closest_train_basins_df[closest_train_basins_df['ClosestTrainBasin'] == str(file_num)]['TestBasin'].tolist()
        print(f"test_basins: {test_basins}")
        log_file.write(f"train_basins: {train_basins}\n")
        log_file.write(f"test_basins: {test_basins}\n")
        log_file.flush()
        
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

        bma_weights = bma_weights_og[f'file_{file_num}'].values

        ##########!!!!
        # CALIBRATION
        # precip_cal: USED TO TRAIN THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_cal:  EXPECTED OUTPUT OF THE TRAIN DATA, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        # EVALUATION
        # precip_eva:  INPUT OF THE ESN , EX; PRECIPITATION, TEMPERATURE, OTHER INPUTS
        # obs_eva: EVALUATE THE TEST OBSERVED, SHIFTED ONE TIMESTEP OF TRAIN DATA TO PREDICT,EX: OBSERVED FLOW

        W_out, reservoir = model.train(input_train_cal, target_data=error_train_target_cal, washout=washout, ridge_param=ridge_param)

        W_out_weights = W_out.copy()
        W_out_row = np.concatenate(([f'file_{file_num}'], W_out.flatten()))
        W_out_rows.append(W_out_row)

        # W_out (1,R)
        # reservoir (R, )

        #W_out_df = pd.DataFrame(W_out)
        #W_out_df.to_csv(f"out/{loc}/BcProx/W_out_file_{file_num}.csv", index=False)
        error_train_cal = model.predict(reservoir, input_train_cal, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
        error_train_eva = model.predict(reservoir, input_train_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)

        train_predict_cal = error_train_cal + np.concatenate([df_train_cal['Sim flow'].values[1:], df_train_eva['Sim flow'].values[0:1]])
        train_predict_eva = error_train_eva + df_train_eva['Sim flow'].values[1:]

        train_predict_cal[train_predict_cal < 0] = 0
        train_predict_eva[train_predict_eva < 0] = 0    

        train_file_row_cal = [f'file_{file_num}_cal'] + list(train_predict_cal.flatten())
        train_file_row_eva = [f'file_{file_num}_eva'] + list(train_predict_eva.flatten())

        if file_num == train_basins[0]:
            train_predict_cal_dates = pd.date_range(start=pd.to_datetime(start_date_cal) + pd.DateOffset(days=1), periods=len(train_predict_cal[0]))
            train_predict_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(train_predict_eva[0]))

            train_date_row_cal = list(['Date'] + [str(date.date()) for date in train_predict_cal_dates])
            train_date_row_eva = list(['Date'] + [str(date.date()) for date in train_predict_eva_dates])

            train_predict_cal_df = pd.DataFrame([train_date_row_cal])
            train_predict_eva_df = pd.DataFrame([train_date_row_eva])

            train_predict_cal_df.to_csv(output_dir + f"/train_basin/predict/BcProx_predict{file_tag}_cal.csv", mode='w', index=False, header=False)
            train_predict_eva_df.to_csv(output_dir + f"/train_basin/predict/BcProx_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

        with open(output_dir + f"/train_basin/predict/BcProx_predict{file_tag}_cal.csv", 'a') as file:
            pd.DataFrame([train_file_row_cal]).to_csv(file, header=False, index=False)
        with open(output_dir + f"/train_basin/predict/BcProx_predict{file_tag}_eva.csv", 'a') as file:
            pd.DataFrame([train_file_row_eva]).to_csv(file, header=False, index=False)

        file_results_train_cal = {'file_num': file_num}
        file_results_train_eva = {'file_num': file_num}

        for benchmark in benchmark_list:
            file_results_train_cal.update({
                f'BcProx{file_tag}_{benchmark}_cal': BMK(target_train_cal, train_predict_cal, benchmark)
            })
            file_results_train_eva.update({
                f'BcProx{file_tag}_{benchmark}_eva': BMK(target_train_eva, train_predict_eva, benchmark)
            })

        results_train_cal.append(file_results_train_cal)
        results_train_eva.append(file_results_train_eva)

        del W_out, train_predict_eva, train_predict_cal,  error_train_cal, error_train_eva

        ### TEST BASINS ###
        for predict_file_num in test_basins:
            df_test = load_data(predict_file_num, varssim_dir, start_date_cal, end_date_eva, loc) # to get the observed flow for benchmarking

            df_test_eva = df_test[start_date_eva:end_date_eva].copy()

            input_test_eva = np.hstack([
                df_test_eva['Precip'].values.reshape(-1, 1),
                df_test_eva['Temp'].values.reshape(-1, 1),
                df_test_eva['PET'].values.reshape(-1, 1)]).T # Observed - Simulation
            input_test_eva = input_test_eva[:, :-1]

            # get BMA values using the weights of training basin
            df_BMA = np.zeros(len(df_test_eva))  

            #print(bma_weights)
            for date in df_test_eva.index:
                BMA_day_prediction = 0
                for i, model_num in enumerate(model_list):
                    model_eva = df_test_eva.at[date, model_num]
                    BMA_day_prediction += bma_weights[i] * model_eva
                df_BMA[df_test_eva.index.get_loc(date)] = BMA_day_prediction

            df_test_eva['Sim flow'] = df_BMA

            # もとの入力値のtimestep = t のとき、error_cal&evaのtimestep = t+1 になるようにしている
            # predict = error_cal + BMAsim
            # predict のtimestep = t+1
            error_test_eva = model.predict_PUB(W_out_weights, input_test_eva, ptb_func=None, ptb_scale=1.0, nexttime=nexttime, extended_interval=10)
            predict_test_eva = error_test_eva + df_test_eva['Sim flow'].values[1:]
            predict_test_eva[predict_test_eva < 0] = 0  

            test_file_row_eva = [f'file_{predict_file_num}_eva'] + list(predict_test_eva.flatten())

            if predict_file_num == test_basins[0] and file_num == train_basins[0]:
                predict_test_eva_dates = pd.date_range(start=pd.to_datetime(start_date_eva) + pd.DateOffset(days=1), periods=len(predict_test_eva[0]))

                test_date_row_eva = list(['Date'] + [str(date.date()) for date in predict_test_eva_dates])
                test_predict_eva_df = pd.DataFrame([test_date_row_eva])

                test_predict_eva_df.to_csv(output_dir + f"/test_basin/predict/BcProx_predict{file_tag}_eva.csv", mode='w', index=False, header=False)

            with open(output_dir + f"/test_basin/predict/BcProx_predict{file_tag}_eva.csv", 'a') as file:
                pd.DataFrame([test_file_row_eva]).to_csv(file, header=False, index=False)

            file_results_test_eva = {'file_num': predict_file_num}
            target_test_eva = np.concatenate((df_test_eva['Obs flow'].values[1:],)) #1089
            for benchmark in benchmark_list:
                file_results_test_eva.update({
                    f'BcProx{file_tag}_{benchmark}_eva': BMK(target_test_eva, predict_test_eva, benchmark)
                })  

            results_test_eva.append(file_results_test_eva)

        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")

        del reservoir

    log_file.write(f"DONE\n")
    log_file.close()

    W_out_df = pd.DataFrame(W_out_rows)
    W_out_df.to_csv(output_dir + f"/train_basin/Wout/BcProx_W_out{file_tag}.csv", index=False, header=False)

    ### ONLY FOR TRAINING BASINS ###
    df_results_train_cal = pd.DataFrame(results_train_cal)
    df_results_train_cal.to_csv(output_dir + f'/train_basin/results/BcProx_results{file_tag}_cal.csv', index=False)
    df_results_train_eva = pd.DataFrame(results_train_eva)
    df_results_train_eva.to_csv(output_dir + f'/train_basin/results/BcProx_results{file_tag}_eva.csv', index=False)

    ### FOR TEST BASINS ###
    df_results_test_eva = pd.DataFrame(results_test_eva)
    df_results_test_eva.to_csv(output_dir + f'/test_basin/results/BcProx_results{file_tag}_eva.csv', index=False)

    print(f"BcProx{file_tag} is done!")
print("DONE!!!")