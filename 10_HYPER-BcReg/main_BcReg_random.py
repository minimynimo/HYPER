# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
# selects the training basin at random.  The selection is to be as evenly to chosen as possible(create equaly parted sections to select one value)
import matplotlib
matplotlib.use('Agg')
from esn_BcReg import ESN
from run_BcReg_random import weight_vector, run_BC_pre_PCA, PCA_lasso, run_BC_post_PCA, BayesianModelAveraging
import os
import pandas as pd 
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

# NOTE might need to do standard scaler to the attributes before lasso
# doing standard scaler helps with a more realistic coefficient values
# but the PCA revertion results without reverting the standardization does not give good results (see fig)
# I think we need to revert the standardization on predictions ? maybe
# or rethink the way the attributes are standardized

## Flow
# 1. Run BC model with 700 reservoirs
# 2. PCA on the weights of the BMA and RC models
# 3. Use lasso regression to revert PCA weights
# 4. Using the reverted weights, run the BC model again

#####################
n_components = 3
alpha = 0.1
fs = 15
benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]
###

nexttime = True

buf = ""

#loc, ver = "JP", 1
loc, ver = "JP", 2


input_size = 3
output_size = 1

spectral_radius = 0.4
washout = 0
ridge_param = 1.0 

BMA_data_exist = True

random_samples = 100
random_seed = 42 

####################

ver_name = "ver1_1" if ver == 1 else "ver2_0" 
attribute_dir = f'hyper/data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    start_val_list = [68]
    #step_val_list = list(range(2,69))
    step_val_list = [2,4,8,15,20,25,30,40,50,68]
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]# chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    train_basin_int_list = [70,50,30,20,15,10,5,3]

    basin_data_df = pd.read_csv("hyper/data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','land_GolfCourse','land_GolfCourse_Ratio']


varssim_dir = f"hyper/data/MERVJP/varssim_nocal/{ver_name}"

file_tag = f"_r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}{buf}"
#####################
print(file_tag)

file_list = list(range(1, file_tot_num+1))

# possible training basins to select the training basins from
PosTrainBasin = [i for i in file_list if i not in test_basins_list]

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

# 1. Run BC model with 700 reservoirs
model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)

output_base_dir = f'hyper/out/{loc}/BcReg_random_{reservoir_size}_{ridge_param}'
os.makedirs(output_base_dir, exist_ok=True)
if os.path.exists(f'{output_base_dir}/BcReg_random{file_tag}_log.txt'):
    open(f'{output_base_dir}/BcReg_random{file_tag}_log.txt', 'w').close()
log_file = open(f'{output_base_dir}/BcReg_random{file_tag}_log.txt', 'a')

param_file_path = os.path.join(output_base_dir, 'parameters.txt')
with open(param_file_path, 'w') as param_file:
    param_file.write("\n===RC===\n")
    param_file.write(f"reservoir_size = {reservoir_size}\n")
    param_file.write(f"spectral_radius = {spectral_radius}\n")
    param_file.write(f"ridge_param = {ridge_param}\n")
    param_file.write("\n===PCA===\n")
    param_file.write(f"alpha = {alpha}\n")

# train everything first to get the weights
#### BC TRAIN ####
# BMA weights
if BMA_data_exist:
    bma_weights_df = pd.read_csv(f"hyper/out/{loc}/BMA/weights/BMA_weights.csv", index_col=0, header=0)
    bma_predict_cal_df = pd.read_csv(f"hyper/out/{loc}/BMA/predict/BMA_predict_cal.csv", index_col=0, header=0)
    bma_predict_eva_df = pd.read_csv(f"hyper/out/{loc}/BMA/predict/BMA_predict_eva.csv", index_col=0, header=0)

    bma_weights_df = bma_weights_df.to_dict(orient='index')
    bma_predict_cal_df = bma_predict_cal_df.to_dict(orient='index')
    bma_predict_eva_df = bma_predict_eva_df.to_dict(orient='index')

    bma_weights_og = {f'file_{train_basin}': bma_weights_df[f'file_{train_basin}'] for train_basin in PosTrainBasin}
    bma_df_cal_og = {f'file_{train_basin}_cal': np.array(list(bma_predict_cal_df[f'file_{train_basin}_cal'].values())) for train_basin in PosTrainBasin}
    bma_df_eva_og = {f'file_{train_basin}_eva': np.array(list(bma_predict_eva_df[f'file_{train_basin}_eva'].values())) for train_basin in PosTrainBasin}
else:
    bma_weights_og, bma_df_cal_og, bma_df_eva_og = BayesianModelAveraging(PosTrainBasin, model_list, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc)

bma_weights_og = pd.DataFrame(bma_weights_og)
bma_df_cal_og = pd.DataFrame(bma_df_cal_og)
bma_df_eva_og = pd.DataFrame(bma_df_eva_og)

bma_cal_og = bma_df_cal_og[[f'file_{num}_cal' for num in PosTrainBasin]]
bma_eva_og = bma_df_eva_og[[f'file_{num}_eva' for num in PosTrainBasin]]
bma_weights_og = bma_weights_og[[f'file_{num}' for num in PosTrainBasin]]

result_cal_og, result_eva_og, W_out_og_rows = run_BC_pre_PCA(model, 
                                                        PosTrainBasin,
                                                        loc,
                                                        output_base_dir,
                                                        bma_cal_og, 
                                                        bma_eva_og, 
                                                        start_date_cal, 
                                                        end_date_cal, 
                                                        start_date_eva, 
                                                        end_date_eva, 
                                                        benchmark_list, 
                                                        file_tag, 
                                                        washout, 
                                                        ridge_param, 
                                                        nexttime, 
                                                        varssim_dir)

for train_basin_int in train_basin_int_list:
    log_file.write(f"Train Basin Int: {train_basin_int}\n")
    log_file.flush()

    output_dir = f'{output_base_dir}/Train{train_basin_int}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/W_out', exist_ok=True)
    #os.makedirs(output_dir + '/reservoir', exist_ok=True)
    os.makedirs(output_dir + '/PCA', exist_ok=True)

    random.seed(random_seed)

    for sample in range(1, random_samples+1):
        file_tag = f"_Train{train_basin_int}_sample{sample}"
        print(file_tag)

        start_time = datetime.now()
        start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"Train: {train_basin_int}, Sample: {sample}\n")
        log_file.write(f"start: {start_time_st}\n")
        log_file.flush()

        output_fig_dir = f'hyper/fig/{loc}/BcReg_random_{reservoir_size}_{ridge_param}/Train{train_basin_int}/sample{sample}'
        os.makedirs(output_fig_dir, exist_ok=True)
        #for subdir in [f'train_basin/results/sample{sample}', f'train_basin/predict/sample{sample}', f'train_basin/reservoir/sample{sample}']:
        #    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        for subdir in [f'test_basin/results/sample{sample}', f'test_basin/predict/sample{sample}', f'test_basin/reservoir/sample{sample}']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        train_basins = random.sample(PosTrainBasin, train_basin_int) # randomly select train basins
        train_basins = sorted(train_basins)
        print(f"Train basins: {train_basins}")

        #log_file.write("BMA\n")
        #log_file.flush()

        bma_weights = bma_weights_og[[f'file_{train_basin}' for train_basin in train_basins]]

        #result_cal_og = result_cal_og[[f'file_{train_basin}_cal' for train_basin in train_basins]]
        #result_eva_og = result_eva_og[[f'file_{train_basin}_eva' for train_basin in train_basins]]

        #df_results_cal_og = pd.DataFrame(result_cal_og)
        #df_results_cal_og.to_csv(output_dir + f'/train_basin/results/sample{sample}/BcReg_results{file_tag}_cal.csv', index=False)

        #df_results_eva_og = pd.DataFrame(result_eva_og)
        #df_results_eva_og.to_csv(output_dir + f'/train_basin/results/sample{sample}/BcReg_results{file_tag}_eva.csv', index=False)

        W_out_og_df = pd.DataFrame(W_out_og_rows)
        W_out_og_df = W_out_og_df[W_out_og_df[0].isin([f'file_{train_basin}' for train_basin in train_basins])]

        W_out_og_df.set_index(0, inplace=True)
        W_out_og_df.to_csv(output_dir + f'/W_out/BcReg_W_out{file_tag}_bma.csv', header=False)

        print(f"BC TRAINING DONE")
        #log_file.write(f"BC TRAINING DONE\n")

        #### PCA ####
        #log_file.write(f"PCA\n")
        #log_file.flush()
        X = weight_vector(bma_weights, W_out_og_df, reservoir_size, model_list, train_basins).values # (file_num, 744)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_components > len(train_basins):
            n_components = len(train_basins)
        pca = PCA(n_components=n_components)
        pc_values = pca.fit_transform(X_scaled) # primary components (PC1~3) of each basins (135, 3)
        pc_values = pd.DataFrame(pc_values, columns=[f'PC{i}' for i in range(1, n_components + 1)])
        pc_values.index = [f'{i}' for i in train_basins]
        pc_values.index.name = 'file_num'

        #### LASSO ####
        #log_file.write(f"PCA lasso\n")
        #log_file.flush()

        output_dir_PCA = output_dir + '/PCA'
        os.makedirs(output_dir_PCA, exist_ok=True)  
        os.makedirs(output_dir_PCA + '/predicted_pcs_train', exist_ok=True)
        os.makedirs(output_dir_PCA + '/predicted_pcs_test', exist_ok=True)  

        # pc_values: only for training basins, attribute_values: all basins
        lasso_model, predicted_pcs_train, predicted_pcs_test = PCA_lasso(pc_values, attribute_values, train_basins, test_basins_list, n_components, alpha, fs, output_dir_PCA, output_fig_dir, columns_drop, sample = sample)
        predicted_pcs_train_df = pd.DataFrame(predicted_pcs_train, columns=[f'PC{i}' for i in range(1, n_components + 1)])
        predicted_pcs_train_df.index = [f'file_{file_num}' for file_num in train_basins]
        predicted_pcs_train_df.to_csv(output_dir_PCA + f'/predicted_pcs_train/PCA_lasso_predicted_pcs_sample{sample}_train.csv') # (file num, PCs)

        predicted_pcs_test_df = pd.DataFrame(predicted_pcs_test, columns=[f'PC{i}' for i in range(1, n_components + 1)])
        predicted_pcs_test_df.index = [f'file_{file_num}' for file_num in test_basins_list]
        predicted_pcs_test_df.to_csv(output_dir_PCA + f'/predicted_pcs_test/PCA_lasso_predicted_pcs_sample{sample}_test.csv') # (file num, PCs)

        ### BcReg ###
        for PC_n in range(1, n_components + 1): #1~5
            #log_file.write(f"PC{PC_n}\n")
            #log_file.flush()
            print(f"PC_n {PC_n}")

            ### reverting PCA ###
            for train_test in ['test']: #'train', 
                if train_test == 'train':
                    predicted_pcs_df = predicted_pcs_train_df
                    file_list_test_train = train_basins
                elif train_test == 'test':
                    predicted_pcs_df = predicted_pcs_test_df
                    file_list_test_train = test_basins_list
            
                X_pca = predicted_pcs_df.values[:, :PC_n] # (file_num, n components)
                eigenvector_n = pca.components_[:PC_n] # (n_components, 744)

                X_orig = np.dot(X_pca, eigenvector_n)
                X_orig_backscaled = scaler.inverse_transform(X_orig)
                weights_inverted = pd.DataFrame(X_orig_backscaled, index=[f'file_{file_num}' for file_num in file_list_test_train])

                result_eva_rev, reservoir_rev_rows = run_BC_post_PCA(model, 
                                                                    file_list_test_train, 
                                                                    loc,
                                                                    output_dir,
                                                                    train_test,
                                                                    PC_n, 
                                                                    weights_inverted, 
                                                                    model_list, 
                                                                    start_date_eva, 
                                                                    end_date_eva, 
                                                                    benchmark_list, 
                                                                    nexttime, 
                                                                    file_tag, 
                                                                    varssim_dir,
                                                                    sample = sample)

                df_results_eva_rev = pd.DataFrame(result_eva_rev)
                df_results_eva_rev.to_csv(output_dir + f'/{train_test}_basin/results/sample{sample}/BcReg_results{file_tag}_rev_PC{PC_n}_eva.csv', index=False)

                reservoir_rev_rows_df = pd.DataFrame(reservoir_rev_rows)
                reservoir_rev_rows_df.to_csv(output_dir + f"/{train_test}_basin/reservoir/sample{sample}/BcReg_reservoir{file_tag}_rev_PC{PC_n}.csv", index=False, header=False)

        end_time = datetime.now()
        end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
        log_file.write(f"end: {end_time_st}\n")
        log_file.write(f"elapsed: {end_time - start_time}\n")
        log_file.write("==================================================\n")

    log_file.write(f"Train Basin Int: {train_basin_int} DONE\n")
    log_file.flush()

log_file.write(f"DONE\n")
log_file.close()
print("DONE")