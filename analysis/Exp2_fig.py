import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

fig_size = (6, 6)
fig_size_cdf = (6, 6)
fs = 17 
fs_cdf = 13 

BcProx_path_val = "hyper/out/JP/BcProx_kfold_200_0.001/test_basin/results/BcProx_results_r200_sr0.4_rr0.001_eva.csv"
BcReg_path_val = "hyper/out/JP/BcReg_kfold_200_0.001/test_basin/results/BcReg_results_rev_PC2_eva.csv"
LSTM_path_val = "hyper/out/JP/LSTM_PUB_kfold/results/LSTM_PUB_results_eva.csv"

# Load data
BcProx_data = pd.read_csv(BcProx_path_val)
BcReg_data = pd.read_csv(BcReg_path_val)
LSTM_data = pd.read_csv(LSTM_path_val)

benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]
os.makedirs('hyper/fig/JP/benchmark/kfold/box', exist_ok=True)
os.makedirs('hyper/fig/JP/benchmark/kfold/cdf', exist_ok=True)

benchmark_limits= {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -1, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max':10},
    'MAE': {'min': 0, 'max': 10}
}

# Extract relevant columns (assuming the evaluation metric is in a column named 'value')
for benchmark in benchmark_list:
    BcProx_values = BcProx_data[f'BcProx_r200_sr0.4_rr0.001_{benchmark}_eva']
    BcReg_values = BcReg_data[f'BcReg_r200_sr0.4_rr0.001_{benchmark}_eva']
    LSTM_values = LSTM_data[f'LSTM_PUB_{benchmark}_eva']

    # Combine data for box plot
    data = [BcProx_values, BcReg_values, LSTM_values]
    labels = ['BcProx', 'BcReg', 'LSTM']

    # Create box plot
    plt.figure(figsize=fig_size)
    box = plt.boxplot(data, labels=labels, patch_artist=True)

    # Set colors for each box
    colors = ['#b39ddb', '#e57373', '#7cc7bc']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set median line color to black
    for median in box['medians']:
        median.set_color('black')

    plt.ylabel(f'{benchmark}', fontsize=fs)
    plt.grid(axis='y')
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])


    plt.tight_layout()

    # Save or show the plot
    plt.savefig(f'hyper/fig/JP/benchmark/kfold/box/kfold_results_{benchmark}.png')

    plt.close()

    # Create CDF plot
    colors_cdf = ['#7b3294', '#d7191c', '#2ca02c']  # green, purple, red
    plt.figure(figsize=fig_size_cdf)
    for values, label, color in zip(data, labels, colors_cdf):
        sorted_values = np.sort(values)
        yvals = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, yvals, label=label, color=color, linewidth=1.5)
    plt.xlabel(f'{benchmark}', fontsize=fs_cdf)
    plt.ylabel('CDF', fontsize=fs_cdf)
    plt.grid()
    plt.legend(fontsize=fs_cdf)
    plt.xticks(fontsize=fs_cdf)
    plt.yticks(fontsize=fs_cdf)
    plt.xlim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'hyper/fig/JP/benchmark/kfold/cdf/kfold_results_{benchmark}_cdf.png')
    plt.close()
    print(f"Processed {benchmark} successfully.")




