# Description: Main script for running the ESN model on multiple files.
# can be used for both BMA and model-based bias correction
import matplotlib
matplotlib.use('Agg')
from esn_BC_PCA_lasso_PUB import ESN
from run_BC_PCA_PUB import weight_vector, run_BC_pre_PCA, PCA_lasso, run_BC_post_PCA, BayesianModelAveraging
import os
import pandas as pd 
import numpy as np
from numpy.ma import masked_array
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
spin = 1

nexttime = True
BMA_data_exist = True

non_arid_files = True

buf = ""

#loc, ver = "JP", 1
#loc, ver = "JP", 2
#loc, ver = "US", 1
loc, ver = "US", 2
#loc, ver = "GB", 2

if loc != "US":
    non_arid_files = False

input_size = 3
output_size = 1
#reservoir_size = 700
spectral_radius = 0.4
washout = 0
ridge_param = 1.0

#####################


ver_name = "ver1_1" if ver == 1 else "ver2_0" 
attribute_dir = f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}'

if loc == "JP" and ver == 1:
    file_tot_num  = 135
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_limited_met&soil&geology&land_{ver_name}.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
elif loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    attribute_values = pd.read_csv(f'{attribute_dir}/basin_data_JP_cleaned_limited.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','grdc_no','river','station','lat_org','long_org']
    region_list = ['Hokkaido', 'Tohoku', 'Kanto', 'Tokai-Kinki', 'Shikoku-Kyusyu']
    region_list = ['Tokai-Kinki', 'Shikoku-Kyusyu']
    region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/{loc}/0_data/river_basin/dataset_{loc}/pub_region_list_{ver_name}.csv')
elif loc == "US" and ver == 1:
    file_tot_num = 669
    reservoir_size = 200

    if non_arid_files:
        arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
        arid_file_list = pd.read_csv(arid_file_list)
        arid_file_list = arid_file_list['File_num'].tolist()
        arid_file_list.sort()
        non_arid_file_list = [i for i in range(1,file_tot_num+1) if i not in arid_file_list]

    attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']
elif loc == "US" and ver == 2:
    file_tot_num = 667
    reservoir_size = 200

    if non_arid_files:
        arid_file_list = "/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/arid_file_num.csv"
        arid_file_list = pd.read_csv(arid_file_list)
        arid_file_list = arid_file_list['File_num'].tolist()
        arid_file_list.sort()
        non_arid_file_list = [i for i in range(1,file_tot_num+1) if i not in arid_file_list]

    test_basins_list = list(range(4,file_tot_num,10)) #67 values, going 4,14,24,34,,,
    attribute_values = pd.read_csv(f'{attribute_dir}/camels_attribute_combined.csv', encoding= 'UTF-8') # File_num: 1~135
    columns_drop = ['File_num','gauge_id','gauge_name','country']
    region_list = ['EastCoast_N','EastCoast_C', 'EastCoast_S', 'Inland_N', 'Inland_S', 'WestCoast_N', 'WestCoast_S']
    region_list = ['WestCoast_S']
    region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/US/0_data/river_basin/dataset_US/file_num_region.csv')
elif loc == "GB" and ver == 2:
    file_tot_num = 396
    reservoir_size = 300

    test_basins_list = list(range(4, file_tot_num, 10))
    attribute_values = pd.read_csv(f'{attribute_dir}/camelsgb_attribute_combined.csv', encoding= 'UTF-8')
    columns_drop = ['File_num','gauge_id','gauge_name','country']
    region_list = ["Northernmost", "NorthMid", "NorthEdge", "SouthEdge", "SouthMid", "Southernmost"]
    region_list = ["Southernmost"]
    region_list_df = pd.read_csv(f'/data0/funato/3_gis_data/GB/0_data/river_basin/dataset_GB/file_num_valid_small.csv')
    
varssim_dir = f"/data0/funato/2_MERV/{loc}/varssim_nocal/{ver_name}"
file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}{buf}"
print(file_tag)

if non_arid_files:
    file_list = non_arid_file_list
    non_arid_dir = "non_arid/"
else:
    file_list = list(range(1, file_tot_num+1))


start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

model_list = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

if BMA_data_exist:
    bma_weights_df = pd.read_csv(f"/data0/funato/0_out/99_out/{loc}/BMA/weights/BMA_weights.csv", index_col=0, header=0)
    bma_predict_cal_df = pd.read_csv(f"/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_cal.csv", index_col=0, header=0)
    bma_predict_eva_df = pd.read_csv(f"/data0/funato/0_out/99_out/{loc}/BMA/predict/BMA_predict_eva.csv", index_col=0, header=0)

    bma_weights_df = bma_weights_df.to_dict(orient='index')
    bma_predict_cal_df = bma_predict_cal_df.to_dict(orient='index')
    bma_predict_eva_df = bma_predict_eva_df.to_dict(orient='index')

# 1. Run BC model with 700 reservoirs
model = ESN(input_size=input_size,
            output_size=output_size,
            reservoir_size=reservoir_size,
            adjacency_density=0.0006,
            spectral_radius=spectral_radius,
            input_scale=0.5)


for region in region_list:
    output_dir = f'/data0/funato/0_out/99_out/{loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}/{non_arid_dir}region/{region}'
    output_fig_dir = f'/data0/funato/0_out/0_fig/{loc}/BC-PCA-lasso_PUB_{reservoir_size}_{ridge_param}/{non_arid_dir}region/{region}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/W_out', exist_ok=True)
    os.makedirs(output_dir + '/PCA', exist_ok=True)
    os.makedirs(output_fig_dir, exist_ok=True)
    for subdir in ['train_basin/results', 'train_basin/predict','train_basin/reservoir' ,'test_basin/results', 'test_basin/predict', 'test_basin/reservoir']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    if os.path.exists(output_dir + f'/BC-PCA-lasso_PUB{file_tag}_log.txt'):
        open(output_dir + f'/BC-PCA-lasso_PUB{file_tag}_log.txt', 'w').close()
    log_file = open(output_dir + f'/BC-PCA-lasso_PUB{file_tag}_log.txt', 'a')
    log_file.flush()

    file_tag = f"_r{reservoir_size}_s{spin}_sr{spectral_radius}_rr{ridge_param}"
    print(file_tag)

    file_list_train = sorted(region_list_df[region_list_df['Region'] == region]['File_num'].tolist())
    file_list_train = [file_num for file_num in file_list_train if file_num not in test_basins_list]
    print(f'region: {region}, {len(file_list_train)} basins')
    print(file_list_train)
    file_list_test = [num for num in file_list if num not in file_list_train]

    #### BMA ####
    log_file.write("BMA\n")
    log_file.flush()

    if BMA_data_exist:
        bma_df_cal_og = {f'file_{train_basin}_cal': np.array(list(bma_predict_cal_df[f'file_{train_basin}_cal'].values())) for train_basin in file_list_train}
        bma_df_eva_og = {f'file_{train_basin}_eva': np.array(list(bma_predict_eva_df[f'file_{train_basin}_eva'].values())) for train_basin in file_list_train}
        bma_weights_og = {f'file_{train_basin}': bma_weights_df[f'file_{train_basin}'] for train_basin in file_list_train}
    else:
        bma_weights_og, bma_df_cal_og, bma_df_eva_og = BayesianModelAveraging(model_list, file_list_train, loc, start_date_cal, end_date_cal, start_date_eva, end_date_eva, varssim_dir)

    bma_weights_og = pd.DataFrame(bma_weights_og)
    bma_df_cal_og = pd.DataFrame(bma_df_cal_og)
    bma_df_eva_og = pd.DataFrame(bma_df_eva_og)

    bma_cal_og = bma_df_cal_og[[f'file_{num}_cal' for num in file_list_train]]
    bma_eva_og = bma_df_eva_og[[f'file_{num}_eva' for num in file_list_train]]
    bma_weights_og = bma_weights_og[[f'file_{num}' for num in file_list_train]]

    #### BC TRAIN ####
    result_cal_og, result_eva_og, W_out_og_rows = run_BC_pre_PCA(model, 
                                                                file_list_train,
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
                                                                varssim_dir,
                                                                log_file=log_file)

    W_out_og_df = pd.DataFrame(W_out_og_rows)
    W_out_og_df.set_index(0, inplace=True)
    W_out_og_df.to_csv(output_dir + f'/W_out/BC-PCA-lasso_W_out{file_tag}_bma.csv', header=False)

    df_results_cal_og = pd.DataFrame(result_cal_og)
    df_results_cal_og.to_csv(output_dir + f'/train_basin/results/BC-PCA-lasso_results{file_tag}_cal.csv', index=False)

    df_results_eva_og = pd.DataFrame(result_eva_og)
    df_results_eva_og.to_csv(output_dir + f'/train_basin/results/BC-PCA-lasso_results{file_tag}_eva.csv', index=False)

    print(f"BC TRAINING DONE")
    log_file.write(f"BC TRAINING DONE\n")

    #### PCA ####
    log_file.write(f"PCA\n")
    log_file.flush()
    X = weight_vector(bma_weights_og, W_out_og_df, reservoir_size, model_list, file_list_train).values # (file_num, 744)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_components > len(file_list_train):
        n_components = len(file_list_train)
    pca = PCA(n_components=n_components)
    pc_values = pca.fit_transform(X_scaled) # primary components (PC1~3) of each basins (135, 3)
    pc_values = pd.DataFrame(pc_values, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    pc_values.index = [f'{i}' for i in file_list_train]
    pc_values.index.name = 'file_num'

    #### LASSO ####
    log_file.write(f"PCA lasso\n")
    log_file.flush()

    output_dir_PCA = output_dir + '/PCA'
    # pc_values: only for training basins, attribute_values: all basins
    lasso_model, predicted_pcs_train, predicted_pcs_test = PCA_lasso(pc_values, attribute_values, file_list_train, file_list_test, n_components, alpha, fs, output_dir_PCA, output_fig_dir, columns_drop)
    predicted_pcs_train_df = pd.DataFrame(predicted_pcs_train, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    predicted_pcs_train_df.index = [f'file_{file_num}' for file_num in file_list_train]
    predicted_pcs_train_df.to_csv(output_dir + f'/PCA/PCA_lasso_predicted_pcs_train.csv') # (file num, PCs)

    predicted_pcs_test_df = pd.DataFrame(predicted_pcs_test, columns=[f'PC{i}' for i in range(1, n_components + 1)])
    predicted_pcs_test_df.index = [f'file_{file_num}' for file_num in file_list_test]
    predicted_pcs_test_df.to_csv(output_dir + f'/PCA/PCA_lasso_predicted_pcs_test.csv') # (file num, PCs)

    ### BC-PCA-lasso ###
    for PC_n in range(1, n_components + 1): #1~5
        log_file.write(f"BC-PCA-lasso{file_tag} {PC_n}\n")
        log_file.flush()
        print(f"PC_n {PC_n}")

        ### reverting PCA ###
        for train_test in ['train', 'test']:
            if train_test == 'train':
                predicted_pcs_df = predicted_pcs_train_df
                file_list_test_train = file_list_train
            else:
                predicted_pcs_df = predicted_pcs_test_df
                file_list_test_train = file_list_test
        
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
                                                                log_file=log_file)
                                                                

            df_results_eva_rev = pd.DataFrame(result_eva_rev)
            df_results_eva_rev.to_csv(output_dir + f'/{train_test}_basin/results/BC-PCA-lasso_PUB_results{file_tag}_rev_PC{PC_n}_eva.csv', index=False)

            reservoir_rev_rows_df = pd.DataFrame(reservoir_rev_rows)
            reservoir_rev_rows_df.to_csv(output_dir + f"/{train_test}_basin/reservoir/BC-PCA-lasso_PUB_reservoir{file_tag}_rev_PC{PC_n}.csv", index=False, header=False)

        print(f"BC-PCA-lasso_PUB{file_tag} {PC_n} DONE")
        log_file.write(f"BC-PCA-lasso_PUB{file_tag} {PC_n} DONE\n")

log_file.write(f"DONE\n")
log_file.close()
print("DONE")