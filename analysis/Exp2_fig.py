import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#########
str_stp_tag = False

loc, ver = "JP", 2
nocal_tag = ''
n_components = 3
reservoir_size = 200
#########
file_tag = f'r{reservoir_size}_s1_sr0.4_rr0.001'

ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 1:
    file_tot_num = 135
elif loc == "JP" and ver == 2:
    file_tot_num = 87

fs = 20 # 6 for paper

def load_BcProx_data():
    BcProx_data = f'hyper/out/{loc}/BcProx_kfold_{reservoir_size}/test_basin/results/BcProx_results_r{reservoir_size}_s1_sr0.4_rr0.001_eva.csv'
    BcProx_df = pd.read_csv(BcProx_data)
    BcProx_column_data = np.array(BcProx_df[f'BcProx_r{reservoir_size}_s1_sr0.4_rr0.001_{benchmark}_eva'].tolist())
    return BcProx_column_data

def load_BmaProx_data():
    BmaProx_data = f'hyper/out/{loc}/BmaProx_kfold/test_basin/results/BmaProx_results_eva.csv'
    BmaProx_df = pd.read_csv(BmaProx_data)
    BmaProx_column_data = np.array(BmaProx_df[f'BmaProx_{benchmark}_eva'].tolist())
    return BmaProx_column_data
    
def load_BcReg_data(pc):
    BcReg_result_data_dir = pd.read_csv(f'hyper/out/{loc}/BcReg_kfold_{reservoir_size}/test_basin/results/BcReg_results_rev_PC{pc}_eva.csv')
    BcReg_column_data = BcReg_result_data_dir[f'BC-PCA-lasso_r{reservoir_size}_s1_sr0.4_rr0.001_{benchmark}_eva']
    return BcReg_column_data

def plot_cdf(ax, data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, label=label)

output_dir = f'/data0/funato/0_out/0_fig/{loc}/benchmark/Spatial_PUB{nocal_tag}_kfold_cdf'
os.makedirs(output_dir, exist_ok=True)

calc_mode_to_name = {
    'BmaProx_norm': 'BmaProx',
    'BcProx_norm': 'BcProx',
    'BcReg_norm': 'BcReg'
}

benchmark_limits = {
    'KGE': {'min': 0, 'max': 1},
    'NSE': {'min': 0, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

benchmark_list = ["KGE","NSE","logNSE","E1","VE", "d","RMSE","MAE"]

for benchmark in benchmark_list:
    fig, ax = plt.subplots(figsize=(8, 8))

    for calc_mode, calc_name in calc_mode_to_name.items():
        if calc_mode == 'BmaProx_norm':
            data = load_BmaProx_data()
            plot_cdf(ax, data, 'BMA-PUB')
        elif calc_mode == 'BcProx_norm':
            data = load_BcProx_data()
            plot_cdf(ax, data, 'BC-PUB')
        elif calc_mode == 'BcReg_norm':
            for pc in range(1, n_components + 1):
                data = load_BcReg_data(pc)
                plot_cdf(ax, data, f'BCPCA-PUB_PC{pc}')

    ax.set_title(f'CDF for {benchmark}', fontsize=fs)
    ax.set_xlabel(f'{benchmark}', fontsize=fs)
    ax.set_ylabel('CDF', fontsize=fs)
    ax.legend()
    ax.grid(True)

    ax.set_xlim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
    ax.set_ylim(0, 1)
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

    output_path = os.path.join(output_dir, f'{benchmark}_cdf_eva.jpg')
    fig.savefig(output_path)
    plt.close(fig)

    print(f'{benchmark} CDF DONE!')

print(f'saved to {output_dir}')
print("DONE!")
