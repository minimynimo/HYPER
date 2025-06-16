# Description: Main script for running the ESN model on multiple files.
import matplotlib
matplotlib.use('Agg')
from esn_BcProx import ESN
import pandas as pd
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
import os
import random
import sys
from run_BcProx_random import run_BC, BMK, load_data, BayesianModelAveraging, split

#####################
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

#loc, ver = "JP", 1
loc, ver = "JP", 2

nexttime = True

input_size = 3
output_size = 1
#reservoir_size = 700
spectral_radius = 0.4
washout = 0
ridge_param = 1.0 #0.001 

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

BMA_data_exist = True

random_samples = 100 #00
random_seed = 42 

#####################
ver_name = "ver1_1" if ver == 1 else "ver2_0"
attribute_dir = f'hyper/data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    #train_basin_int_list = [30,20,15,10,5,3,2]
    train_basin_int_list = [70,50,30,20,10,5,3]
    basin_data_df = pd.read_csv("hyper/data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
    distance_matrix = pd.read_csv('/data0/funato/3_gis_data/JP/0_data/distance_matrix_v2_sorted.csv', index_col=0, header=0)
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']

varssim_dir = f"hyper/data/MERVJP/varssim_nocal/{ver_name}"
file_tag = f"_r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}"

file_list = list(range(1, file_tot_num+1))

# possible training basins to select the training basins from
PosTrainBasin = [i for i in file_list if i not in test_basins_list]

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39", 
              "m42", "m43", "m44", "m46"]

# Load the BMA weights and predictions as dictionaries
if BMA_data_exist:
    bma_weights_df = pd.read_csv(f"hyper/out/{loc}/BMA/weights/BMA_weights.csv", index_col=0, header=0)
    bma_df_cal_og = pd.read_csv(f"hyper/out/{loc}/BMA/predict/BMA_predict_cal.csv", index_col=0, header=0)
    bma_df_eva_og = pd.read_csv(f"hyper/out/{loc}/BMA/predict/BMA_predict_eva.csv", index_col=0, header=0)

    bma_weights_df = bma_weights_df.to_dict(orient='index')
    bma_df_cal_og = bma_df_cal_og.to_dict(orient='index')
    bma_df_eva_og = bma_df_eva_og.to_dict(orient='index')

    bma_weights_og = {f'file_{train_basin}': bma_weights_df[f'file_{train_basin}'] for train_basin in PosTrainBasin}
    bma_df_cal_og = {f'file_{train_basin}_cal': np.array(list(bma_df_cal_og[f'file_{train_basin}_cal'].values())) for train_basin in PosTrainBasin}
    bma_df_eva_og = {f'file_{train_basin}_eva': np.array(list(bma_df_eva_og[f'file_{train_basin}_eva'].values())) for train_basin in PosTrainBasin}
else:
    bma_weights_og, bma_df_cal_og, bma_df_eva_og = BayesianModelAveraging(PosTrainBasin, model_list, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc)

bma_weights_og = pd.DataFrame(bma_weights_og)
bma_df_cal_og = pd.DataFrame(bma_df_cal_og)
bma_df_eva_og = pd.DataFrame(bma_df_eva_og)

model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)

if non_arid_files:
    output_base_dir = f'hyper/out/{loc}/BcProx_random_non_distributed_{reservoir_size}_{ridge_param}_non_arid'
else:
    output_base_dir = f'hyper/out/{loc}/BcProx_random_non_distributed_{reservoir_size}_{ridge_param}'
os.makedirs(output_base_dir, exist_ok=True)
if os.path.exists(output_base_dir + f'/BcProx_random_log.txt'):
    open(output_base_dir + f'/BcProx_random_log.txt', 'w').close()
log_file = open(output_base_dir + f'/BcProx_random_log.txt', 'a')

param_file_path = os.path.join(output_base_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")

result_cal_og, result_eva_og, W_out_weights_og = run_BC(model, PosTrainBasin, bma_df_cal_og, bma_df_eva_og, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc, output_base_dir, washout, ridge_param, nexttime, benchmark_list)
W_out_weights_og = pd.DataFrame(W_out_weights_og)

for train_basin_int in train_basin_int_list:
    log_file.write(f"Train Basin Int: {train_basin_int}\n")
    log_file.flush()

    output_dir = f'{output_base_dir}/Train{train_basin_int}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir in ['train_basin/results', 'train_basin/predict',  'train_basin/Wout']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    for subdir in ['test_basin/results', 'test_basin/predict']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    random.seed(random_seed)

    for sample in range(1, random_samples+1):
        file_tag = f"_Train{train_basin_int}_sample{sample}"
        print(file_tag)

        start_time = datetime.now()
        start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"Train: {train_basin_int}, Sample: {sample}\n")
        log_file.write(f"start: {start_time_st}\n")
        log_file.flush()

        train_basins = random.sample(PosTrainBasin, train_basin_int)
        train_basins = sorted(train_basins)

        # Convert train_basins to strings
        train_basins_str = list(map(str, train_basins))
        print("Train Basins: ", train_basins_str)

        # Create a dataframe to store the closest train basin for each test basin
        closest_train_basins_df = pd.DataFrame(columns=['TestBasin', 'ClosestTrainBasin'])

        # Update the closest_train_basins_df for the current sample
        for test_basin in test_basins_list:
            closest_train_basin = distance_matrix.loc[test_basin, train_basins_str].idxmin()
            new_row = pd.DataFrame([{'TestBasin': test_basin, 'ClosestTrainBasin': closest_train_basin}])
            closest_train_basins_df = pd.concat([closest_train_basins_df, new_row], ignore_index=True)

        #results_train_cal = []
        #results_train_eva = []
        results_test_eva = []

        first_test_basin = True
        for train_file_num in train_basins: # go through only the training basins
            #print("training: ", train_file_num)
            
            test_basins = closest_train_basins_df[closest_train_basins_df['ClosestTrainBasin'] == str(train_file_num)]['TestBasin'].tolist()

            if not test_basins:
                continue

            bma_weights = bma_weights_og[f'file_{train_file_num}'].values

            W_out_weights = W_out_weights_og[W_out_weights_og[0].astype(str) == f'file_{train_file_num}']
            W_out_weights = W_out_weights.iloc[:, 1:].values.flatten().astype(float).tolist()
            W_out_weights = np.array(W_out_weights).reshape(1, -1)

            ### TEST BASINS ###
            for predict_file_num in test_basins:
                #print("testing: ", predict_file_num)
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

                if predict_file_num == test_basins[0] and first_test_basin:
                    first_test_basin = False
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

        ### ONLY FOR TRAINING BASINS ###
        #df_results_train_cal = pd.DataFrame(results_train_cal)
        #df_results_train_cal.to_csv(output_dir + f'/train_basin/results/BcProx_results{file_tag}_cal.csv', index=False)
        #df_results_train_eva = pd.DataFrame(results_train_eva)
        #df_results_train_eva.to_csv(output_dir + f'/train_basin/results/BcProx_results{file_tag}_eva.csv', index=False)

        ### FOR TEST BASINS ###
        df_results_test_eva = pd.DataFrame(results_test_eva)
        df_results_test_eva.to_csv(output_dir + f'/test_basin/results/BcProx_results{file_tag}_eva.csv', index=False)

        print(f"BcProx{file_tag} is done!")

        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")

log_file.write(f"DONE\n")
log_file.close()

print("DONE!!!")