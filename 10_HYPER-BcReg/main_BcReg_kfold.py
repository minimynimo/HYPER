# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
# selects the training basin at random.  The selection is to be as evenly to chosen as possible(create equaly parted sections to select one value)
import matplotlib
matplotlib.use('Agg')
from esn_BC_PCA_lasso_PUB import ESN
from run_BC_PCA_PUB_kfold import weight_vector, run_BC_pre_PCA, PCA_lasso, run_BC_post_PCA, BayesianModelAveraging
import os
import pandas as pd 
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
from sklearn.model_selection import KFold

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
benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]
###

spin = 1

nexttime = True

buf = ""

#loc, ver = "JP", 1
loc, ver = "JP", 2
#loc, ver = "US", 1
#loc, ver = "US", 2
#loc, ver = "GB", 2

kfold_val = 12

input_size = 3
output_size = 1
#reservoir_size = 700
reservoir_size = 200
spectral_radius = 0.4
washout = 0
ridge_param = 0.001 

BMA_data_exist = True
BC_data_exist = False

random_samples = 100
random_seed = 42 

file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
#####################
print(file_tag)

ver_name = "ver1_1" if ver == 1 else "ver2_0" 
attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','land_GolfCourse','land_GolfCourse_Ratio']
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    basin_data_df = pd.read_csv("/data0/funato/3_gis_data/JP/0_data/river_basin/dataset_JP/pub_region_list_ver2_0.csv")
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org','WaterArea','ForestArea','ForestAreaRatio','WaterAreaRatio','land_GolfCourse','land_GolfCourse_Ratio']
elif loc == "US" and ver == 1:
    file_tot_num = 671
    attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']
elif loc == "US" and ver == 2:
    file_tot_num = 667
    basin_data_df = pd.read_csv("/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/pub_region_list_ver2_0.csv")
    attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']
elif loc == "GB" and ver == 2:
    file_tot_num = 396
    basin_data_df = pd.read_csv(f"/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/file_num_valid_small.csv")
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']


varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"

# possible training basins to select the training basins from
PosTrainBasin = list(range(1, file_tot_num + 1))

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

output_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_kfold_{reservoir_size}_{ridge_param}'
os.makedirs(output_dir, exist_ok=True)
if os.path.exists(f'{output_dir}/BC-PCA-lasso_PUB_kfold_log.txt'):
    open(f'{output_dir}/BC-PCA-lasso_PUB_kfold_log.txt', 'w').close()
log_file = open(f'{output_dir}/BC-PCA-lasso_PUB_kfold_log.txt', 'a')

start_time = datetime.now()
start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
log_file.write(f"start: {start_time_st}\n")

# train everything first to get the weights
#### BC TRAIN ####
# BMA weights
if BMA_data_exist:
    bma_weights_df = pd.read_csv(f"/data0/funato/0_out/99_out/{loc}/BMA/weights/BMA_weights.csv", index_col=0, header=0)
    bma_predict_cal_df = pd.read_csv(f"/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_cal.csv", index_col=0, header=0)
    bma_predict_eva_df = pd.read_csv(f"/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_eva.csv", index_col=0, header=0)

    bma_weights_df = bma_weights_df.to_dict(orient='index')
    bma_predict_cal_df = bma_predict_cal_df.to_dict(orient='index')
    bma_predict_eva_df = bma_predict_eva_df.to_dict(orient='index')

    bma_weights_og = {f'file_{train_basin}': bma_weights_df[f'file_{train_basin}'] for train_basin in PosTrainBasin}
    bma_df_cal_og = {f'file_{train_basin}_cal': np.array(list(bma_predict_cal_df[f'file_{train_basin}_cal'].values())) for train_basin in PosTrainBasin}
    bma_df_eva_og = {f'file_{train_basin}_eva': np.array(list(bma_predict_eva_df[f'file_{train_basin}_eva'].values())) for train_basin in PosTrainBasin}
else:
    bma_weights_og, bma_df_cal_og, bma_df_eva_og = BayesianModelAveraging(PosTrainBasin, model_list, varssim_dir, start_date_cal, end_date_cal, start_date_eva, end_date_eva, loc)

bma_weights_og = pd.DataFrame(bma_weights_og)
bma_cal_og = pd.DataFrame(bma_df_cal_og)
bma_eva_og = pd.DataFrame(bma_df_eva_og)

#bma_cal_og = bma_df_cal_og[[f'file_{num}_cal' for num in PosTrainBasin]]
#bma_eva_og = bma_df_eva_og[[f'file_{num}_eva' for num in PosTrainBasin]]
#bma_weights_og = bma_weights_og[[f'file_{num}' for num in PosTrainBasin]]

if BC_data_exist:
    W_out_og_rows = pd.read_csv(f'/data0/funato/0_out/99_out/{loc}/BC_{reservoir_size}/Wout/BC_Wout_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}_bma.csv', header=None, index_col=0)
    W_out_og_rows = W_out_og_rows.reset_index().values.tolist()
else:
    result_cal_og, result_eva_og, W_out_og_rows = run_BC_pre_PCA(model, 
                                                            PosTrainBasin,
                                                            loc,
                                                            output_dir,
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
                                                            spin, 
                                                            nexttime, 
                                                            varssim_dir)



output_fig_dir = f'/data0/funato/0_out/0_fig/{loc}/BC-PCA-lasso_PUB_kfold_{reservoir_size}_{ridge_param}'
os.makedirs(output_fig_dir, exist_ok=True)
#for subdir in [f'train_basin/results/sample{sample}', f'train_basin/predict/sample{sample}', f'train_basin/reservoir/sample{sample}']:
#    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
for subdir in [f'test_basin/results', f'test_basin/predict', f'test_basin/reservoir']:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

random.seed(random_seed)

kf = KFold(n_splits=kfold_val, shuffle=True, random_state=random_seed)

first_test_basin = True
for fold, (train_index, test_index) in enumerate(kf.split(PosTrainBasin)):
    start_time = datetime.now()
    start_time_st = start_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"Fold: {fold}\n")
    log_file.write(f"start: {start_time_st}\n")
    log_file.flush()

    train_basins = [PosTrainBasin[i] for i in train_index]
    test_basins_list = [PosTrainBasin[i] for i in test_index]

    #log_file.write("BMA\n")
    #log_file.flush()

    # get BMA and Wout values for the training basins
    bma_weights = bma_weights_og[[f'file_{train_basin}' for train_basin in train_basins]]

    W_out_og_df = pd.DataFrame(W_out_og_rows)
    W_out_og_df = W_out_og_df[W_out_og_df[0].isin([f'file_{train_basin}' for train_basin in train_basins])]

    #W_out_og_df.set_index(0, inplace=True)
    #W_out_og_df.to_csv(output_dir + f'/W_out/BC-PCA-lasso_W_out_bma.csv', header=False)

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
    lasso_model, predicted_pcs_train, predicted_pcs_test = PCA_lasso(pc_values, attribute_values, train_basins, test_basins_list, n_components, alpha, fs, output_dir_PCA, output_fig_dir, columns_drop)
    
    predicted_pcs_train_df = pd.DataFrame(predicted_pcs_train, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    predicted_pcs_train_df.index = [f'file_{file_num}' for file_num in train_basins]
    predicted_pcs_train_df.to_csv(output_dir_PCA + f'/predicted_pcs_train/PCA_lasso_predicted_pcs_kfold{fold}_train.csv') # (file num, PCs)

    predicted_pcs_test_df = pd.DataFrame(predicted_pcs_test, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    predicted_pcs_test_df.index = [f'file_{file_num}' for file_num in test_basins_list]
    predicted_pcs_test_df.to_csv(output_dir_PCA + f'/predicted_pcs_test/PCA_lasso_predicted_pcs_kfold{fold}_test.csv') # (file num, PCs)

    ### BC-PCA-lasso ###
    for PC_n in range(1, n_components + 1): #1~5
        first_test_basin = True
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
                                                                fold)

            # Save results and predictions to a single file for each PC_n
            results_file_path = output_dir + f'/{train_test}_basin/results/BC-PCA-lasso_PUB_results_rev_PC{PC_n}_eva.csv'
            if fold == 0:
                df_results_eva_rev = pd.DataFrame(result_eva_rev)
                df_results_eva_rev.to_csv(results_file_path, mode='w', index=False)
            else:
                df_results_eva_rev = pd.DataFrame(result_eva_rev)
                df_results_eva_rev.to_csv(results_file_path, mode='a', index=False, header=False)

    if first_test_basin:
        first_test_basin = False
    end_time = datetime.now()
    end_time_st = end_time.strftime("%a %b %d %I:%M:%S %p JST %Y")
    log_file.write(f"end: {end_time_st}\n")
    log_file.write(f"elapsed: {end_time - start_time}\n")
    log_file.write("==================================================\n")

log_file.write(f"DONE\n")
log_file.close()
print("DONE")