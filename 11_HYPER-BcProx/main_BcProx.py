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
#loc, ver = "US", 1
#loc, ver = "US", 2
#loc, ver = "GB",2

spin = 1
nexttime = True

input_size = 3
output_size = 1
#reservoir_size = 700
reservoir_size = 200
spectral_radius = 0.4
washout = 0
ridge_param = 1.0 #0.001 


start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'
#####################
ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)]
    #step_val_list = list(range(2, start_val_list[0] + 1))
    #step_val_list = [2,4,8,15,20,25,30,40]
    # IF test basins eight; 39, 20, 10, 5, 4, 3, 2
    step_val_list = [2,4,8,10,15,17,30]
    step_val_list = [30]
    # IF test basins seventeen; 35,18,9,7,5,4,3
elif loc == "US" and ver == 1:
    file_tot_num = 671
    start_val_list = [336]
    #step_val_list = list(range(2,337))
    step_val_list = [2,5,10,21,42,84,168,337]
elif loc == "US" and ver == 2:
    file_tot_num = 667
    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
    start_val_list = [int(np.floor((file_tot_num - len(test_basins_list))/2) + 1)] #301
    step_val_list = [2,4,6,12,24,47,84,125,149,168,277] #### Train basin num: 299,150,100,50,25,13,7,5,4,3,3
    step_val_list = [149,168,277]

varssim_dir = f"hyper/data/MERVJP/varssim_nocal/{ver_name}"

PosTrainBasin = [i for i in range(1,file_tot_num) if i not in test_basins_list]

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39", 
              "m42", "m43", "m44", "m46"]


model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)

output_base_dir = f'hyper/out/{loc}/BcProx_{reservoir_size}_{ridge_param}'
os.makedirs(output_base_dir, exist_ok=True)
param_file_path = os.path.join(output_base_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")
    param_file.write(f"spin = {spin}\n")


for start_val in start_val_list:
    output_dir = f'{output_base_dir}/start{start_val}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir in ['train_basin/results', 'train_basin/predict',  'train_basin/Wout' ,'test_basin/results', 'test_basin/predict']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for step_val in step_val_list:
        file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_start{start_val}_step{step_val}"
        print(file_tag)

        closest_basins_file = f'/data0/funato/3_gis_data/{loc}/closest/start{start_val}/closest_basins_start{start_val}_step{step_val}.csv'
        closest_basins_df = pd.read_csv(closest_basins_file)

        results_train_cal = []
        results_train_eva = []
        results_test_eva = []
        W_out_rows = []

        # train_basins is the list of basins to be trained in the certain start and step value
        train_basins = sorted([PosTrainBasin[i] for i in range(start_val, len(PosTrainBasin), step_val)] + 
                              [PosTrainBasin[i] for i in range(start_val - step_val, -1, -step_val)])

        bma_weights_og, bma_df_cal_og, bma_df_eva_og = BayesianModelAveraging(train_basins, model_list, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc)
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

            W_out, reservoir = model.train(input_train_cal, target_data=error_train_target_cal, washout=washout, ridge_param=ridge_param, spinoff = spin)

            W_out_weights = W_out.copy()
            W_out_row = np.concatenate(([f'file_{file_num}'], W_out.flatten()))
            W_out_rows.append(W_out_row)

            """W_out_df = pd.DataFrame([W_out_row])
            if file_num == train_basins[0]:
                W_out_df.to_csv(output_dir + f"/train_basin/Wout/BcProx_W_out{file_tag}.csv", mode='w', index=False, header=False)
            else:
                W_out_df.to_csv(output_dir + f"/train_basin/Wout/BcProx_W_out{file_tag}.csv", mode='a', index=False, header=False)
            """
            # W_out (1,R)
            # reservoir (R, )

            #W_out_df = pd.DataFrame(W_out)
            #W_out_df.to_csv(f"hyper/out/{loc}/BcProx/W_out_file_{file_num}.csv", index=False)
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

            end_train_time = datetime.now()
            end_train_time_st = end_train_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
            log_file.write(f"train end: {end_train_time_st}\n")
            log_file.write(f"train elapsed: {end_train_time - start_time}\n")

            start_test_time = datetime.now()
            start_test_time_st = start_test_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
            log_file.write(f"test start: {start_test_time_st}\n")
            log_file.flush()

            ### TEST BASINS ###
            # Get the list of basins to predict
            test_basins = closest_basins_df[
                (closest_basins_df['ClosestBasin'] == file_num) & 
                (closest_basins_df['Basin'] != file_num)# &
                #(closest_basins_df['Basin'].isin(test_basins_list))
            ]['Basin'].tolist()

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