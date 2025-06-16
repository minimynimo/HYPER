# Creates box, line, and point plots for the Spatial vs Regression PUB benchmark
# The norm for all plots of PUB can be shown for the training and testing data
# The difference between the following can be shown for the training and testing data:
### BcReg and BcProx 

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#########
train_int_tag = False

BmaProx_cdf = False
BC_Prox_cdf = False
LSTM_cdf = False
gauged_BC = True

#loc, ver = "JP", 1
loc, ver = "JP", 2

nocal_tag = ''
n_components = 3
spectral_radius = 0.4

samples_ = 100

fs = 13 #27 for presentation, 10 for paper, 
fs_tick = 16 # 20 for presentation, 16 for paper

# size for norm and dif
x_size = 10 #26
y_size = 6
x_size_cdf =16 #17
y_size_cdf = 8 #12
x_size_sum = 8 #9
y_size_sum = 8 #9
#########


ver_name = "ver1_1" if ver == 1 else "ver2_0"

if loc == "JP" and ver == 2:
    file_tot_num = 87
    reservoir_size = 200

    #test_basins_list = [4,11,24,34,40,45,70,77,84] # chosen sequentially based on the file_num when sorted the river basins by latitude 4,14,24,,,
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    train_basin_int_list = [70, 50,30,20,15,10,5,3]
    train_basin_int_list = [70,50,30,20,10,3]
    ridge_param = 1.0

file_tag = f'r{reservoir_size}_sr{spectral_radius}_rr{ridge_param}'

cal_eva = ['test'] #'train', 

benchmark_list = ["KGE","NSE","E1", "VE", "d","RMSE","MAE"]


def create_cdf_plot(ax, plot_df, benchmark, train_test):
    markers_list = ['D', 'o', 's', '^', 'v', 'x', '*']
    markers = markers_list[:len(train_basin_int_list)]
    
    # Define base colors and gradients
    base_colors = {
        'BMA': '#3CB371',
        'BC': 'royalblue',
        'PCA1': 'crimson',
        'PCA2': 'blue',
        'PCA3': 'green',
        'LSTM': 'orange',
        'Gauged BC': 'Black'   # Dark gray for Gauged BC
    }
    color_gradients = {
        'BMA': plt.cm.Greens,
        'BC': plt.cm.Purples,
        'PCA1': plt.cm.Reds,
        'PCA2': plt.cm.Blues,
        'PCA3': plt.cm.Wistia,
        'LSTM': plt.cm.Greens,
    }
    
    reset_interval = len(train_basin_int_list)
    legend_entries = {'BcProx': [], 'Gauged BC': [], 'BcReg': [], 'LSTM':[]}  # Separate lists for BcProx, Gauged BC, and BcReg

    # Loop through columns in plot_df
    for i, column in enumerate(plot_df.columns):
        sorted_data = np.sort(plot_df[column])  # Ensure data is sorted for CDF
        cdf = np.linspace(0, 1, len(sorted_data))  # Ensure proper CDF scaling
        
        # Identify color key
        if 'Reg' in column and 'PC1' in column:
            color_key = 'PCA1'
        elif 'Reg' in column and 'PC2' in column:
            color_key = 'PCA2'
        elif 'Reg' in column and 'PC3' in column:
            color_key = 'PCA3'
        elif 'Reg' in column:
            color_key = 'PCA1'
        elif 'Bma' in column:
            color_key = 'BMA'
        elif 'Gauged BC' in column:
            color_key = 'Gauged BC'
        elif 'LSTM' in column:
            color_key = 'LSTM'
        else:
            color_key = 'BC'

        # Assign color and marker
        if color_key in color_gradients:
            color = color_gradients[color_key](0.15 + 0.75 * (1 - (i % reset_interval) / reset_interval))
        else:
            color = base_colors[color_key]
        marker = '' if column == 'Gauged BC' else markers[i % len(markers)]

        if column == 'Gauged BC':  
            lw_cdf = 3.5
            linestyle = '-'
        else:
            lw_cdf = 1.5
            linestyle = '-'

        # Plot CDF
        line = ax.plot(sorted_data, cdf, label=column, marker=marker, color=color,
                       markersize=9, linewidth=lw_cdf, linestyle=linestyle)[0]

        # Group legend entries into BcProx, Gauged BC, and BcReg categories
        if 'Prox' in column:
            legend_entries['BcProx'].append((line, column.split()[1]))
        elif 'Gauged BC' in column:
            legend_entries['Gauged BC'].append((line, 'Gauged BC'))
        elif 'Reg' in column:
            legend_entries['BcReg'].append((line, column.split()[1]))
        elif 'LSTM' in column:
            legend_entries['LSTM'].append((line, column.split()[1]))

    
    ax.set_title(f'CDF plot for {benchmark} {train_test}\n', fontsize=fs)
    ax.set_xlabel(f'{benchmark}', fontsize=fs_tick)
    ax.set_ylabel('CDF', fontsize=fs_tick)
    ax.xaxis.set_tick_params(labelsize=fs_tick)
    ax.yaxis.set_tick_params(labelsize=fs_tick)
    ax.grid(True)

    # Create legend for BcProx, then Gauged BC, then BcReg with their respective titles
    handles = []
    labels = []

    # Add BcProx group title
    handles.append(Line2D([0], [0], color='none', label='BcProx', linestyle=''))
    labels.append('BcProx')
    # Add lines for BcProx
    for line, label in legend_entries['BcProx']:
        handles.append(line)
        labels.append(label)
        
    # Add Gauged BC
    for line, label in legend_entries['Gauged BC']:
        handles.append(line)
        labels.append(label)
        
    # Add BcReg group title
    handles.append(Line2D([0], [0], color='none', label='BcReg', linestyle=''))
    labels.append('BcReg')
    # Add lines for BcReg
    for line, label in legend_entries['BcReg']:
        handles.append(line)
        labels.append(label)

    # Add LSTM group title
    handles.append(Line2D([0], [0], color='none', label='LSTM', linestyle=''))
    labels.append('LSTM')
    # Add lines for LSTM
    for line, label in legend_entries['LSTM']:
        handles.append(line)
        labels.append(label)

    # Add legend with the specified order and group titles
    #ax.legend(handles, labels, ncol=3, fontsize=fs * 0.8, loc='upper left')
    ax.legend(handles, labels, ncol=3, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs * 1.3)

def load_BcProx_data(train_basin_int, benchmark):
    BcProx_sum_data = None
    BcProx_all_samples = []  # Store all samples for percentile calculations
    for sample in range(1, samples + 1):
        BcProx_data = f'hyper/out/{loc}/BcProx_random_{reservoir_size}_{ridge_param}/Train{train_basin_int}/test_basin/results/BcProx_results_Train{train_basin_int}_sample{sample}_eva.csv'
        BcProx_df = pd.read_csv(BcProx_data, index_col=0)
        BcProx_df = BcProx_df.sort_values(by='file_num')
        BcProx_df = pd.DataFrame(BcProx_df.values)

        BcProx_all_samples.append(BcProx_df.values)  # Collect data for all samples  (samples, test basins, benchmark)

        if BcProx_sum_data is None:
            BcProx_sum_data = BcProx_df.copy()
        else:
            BcProx_sum_data = BcProx_sum_data.add(BcProx_df, fill_value=0)

    
    # Calculate mean
    BcProx_mean_data = BcProx_sum_data.copy()
    BcProx_mean_data = BcProx_mean_data.divide(samples)
    BcProx_mean_data.columns = benchmark_list
    BcProx_mean_data.index.name = 'file_num'
    BcProx_mean_data = BcProx_mean_data[[benchmark]]
    BcProx_mean_data.index = test_basins_list  # Ensure index is set

    # Calculate 10% and 90% distributions
    BcProx_all_samples = np.array(BcProx_all_samples)  # Convert to numpy array for percentile calculations
    BcProx_10th_data = np.percentile(BcProx_all_samples, 10, axis=0)
    BcProx_90th_data = np.percentile(BcProx_all_samples, 90, axis=0)

    BcProx_10th_data = pd.DataFrame(BcProx_10th_data, columns=benchmark_list, index=test_basins_list)
    BcProx_90th_data = pd.DataFrame(BcProx_90th_data, columns=benchmark_list, index=test_basins_list)

    BcProx_10th_data = BcProx_10th_data[[benchmark]]
    BcProx_90th_data = BcProx_90th_data[[benchmark]]

    return BcProx_mean_data, BcProx_10th_data, BcProx_90th_data

def load_BmaProx_data(train_basin_int, benchmark):
    BmaProx_sum_data = None
    BmaProx_all_samples = []  # Store all samples for percentile calculations
    for sample in range(1, samples + 1):
        BmaProx_data = f'hyper/out/{loc}/BmaProx_random/Train{train_basin_int}/test_basin/results/BmaProx_results_Train{train_basin_int}_sample{sample}_eva.csv'
        BmaProx_df = pd.read_csv(BmaProx_data, index_col=0)
        BmaProx_df = BmaProx_df.sort_values(by='file_num')
        BmaProx_df = pd.DataFrame(BmaProx_df.values)

        BmaProx_all_samples.append(BmaProx_df.values)  # Collect data for all samples  (samples, test basins, benchmark)

        if BmaProx_sum_data is None:
            BmaProx_sum_data = BmaProx_df.copy()
        else:
            BmaProx_sum_data = BmaProx_sum_data.add(BmaProx_df, fill_value=0)
    
    BmaProx_mean_data = BmaProx_sum_data.copy()
    BmaProx_mean_data = BmaProx_mean_data.divide(samples)
    BmaProx_mean_data.columns = benchmark_list
    BmaProx_mean_data.index.name = 'file_num'
    BmaProx_mean_data = BmaProx_mean_data[[benchmark]]
    BmaProx_mean_data.index = test_basins_list  # Ensure index is set

    # Calculate 10% and 90% distributions
    BmaProx_all_samples = np.array(BmaProx_all_samples)  # Convert to numpy array for percentile calculations
    BmaProx_10th_data = np.percentile(BmaProx_all_samples, 10, axis=0)
    BmaProx_90th_data = np.percentile(BmaProx_all_samples, 90, axis=0)
    BmaProx_10th_data = pd.DataFrame(BmaProx_10th_data, columns=benchmark_list, index=test_basins_list)
    BmaProx_90th_data = pd.DataFrame(BmaProx_90th_data, columns=benchmark_list, index=test_basins_list)
    BmaProx_10th_data = BmaProx_10th_data[[benchmark]]
    BmaProx_90th_data = BmaProx_90th_data[[benchmark]]

    return BmaProx_mean_data, BmaProx_10th_data, BmaProx_90th_data
    
def load_BcReg_data(pc, train_basin_int, benchmark):
    BcReg_sum_data = None
    BcReg_all_samples = []  
    for sample in range(1, samples + 1):
        BcReg_data = f'hyper/out/{loc}/BcReg_random_{reservoir_size}_{ridge_param}/Train{train_basin_int}/test_basin/results/sample{sample}/BcReg_results_Train{train_basin_int}_sample{sample}_rev_PC{pc}_eva.csv'
        BcReg_df = pd.read_csv(BcReg_data, index_col=0)
        BcReg_df = BcReg_df.sort_values(by='file_num')
        BcReg_df = pd.DataFrame(BcReg_df.values)

        BcReg_all_samples.append(BcReg_df.values)  # Collect data for all samples  (samples, test basins, benchmark)

        if BcReg_sum_data is None:
            BcReg_sum_data = BcReg_df.copy()
        else:
            BcReg_sum_data = BcReg_sum_data.add(BcReg_df, fill_value=0)
    
    BcReg_mean_data = BcReg_sum_data.copy()
    BcReg_mean_data = BcReg_mean_data.divide(samples)
    BcReg_mean_data.index = test_basins_list
    BcReg_mean_data.columns = benchmark_list
    BcReg_mean_data.index.name = 'file_num'
    BcReg_mean_data = BcReg_mean_data[[benchmark]]
    BcReg_mean_data.index = test_basins_list  # Ensure index is set

    # Calculate 10% and 90% distributions
    BcReg_all_samples = np.array(BcReg_all_samples)  # Convert to numpy array for percentile calculations
    BcReg_10th_data = np.percentile(BcReg_all_samples, 10, axis=0)
    BcReg_90th_data = np.percentile(BcReg_all_samples, 90, axis=0)
    BcReg_10th_data = pd.DataFrame(BcReg_10th_data, columns=benchmark_list, index=test_basins_list)
    BcReg_90th_data = pd.DataFrame(BcReg_90th_data, columns=benchmark_list, index=test_basins_list)
    BcReg_10th_data = BcReg_10th_data[[benchmark]]
    BcReg_90th_data = BcReg_90th_data[[benchmark]]

    return BcReg_mean_data, BcReg_10th_data, BcReg_90th_data

def load_LSTM_PUB_data(train_basin_int, benchmark):
    if not LSTM_cdf:
        return None, None, None  # Return None when LSTM_cdf is False
    LSTM_PUB_sum_data = None
    LSTM_PUB_all_samples = []  # Store all samples for percentile calculations
    for sample in range(1, samples + 1):
        LSTM_PUB_data = f'hyper/out/{loc}/LSTM_PUB_random/ensemble/Train{train_basin_int}/test_basin/results/LSTM_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
        LSTM_PUB_df = pd.read_csv(LSTM_PUB_data, index_col=0) # the data of the test basin for the sample
        LSTM_PUB_df = LSTM_PUB_df.sort_values(by='file_num')
        LSTM_PUB_df = pd.DataFrame(LSTM_PUB_df.values)

        LSTM_PUB_all_samples.append(LSTM_PUB_df.values)  # Collect data for all samples  (samples, test basins, benchmark)

        if LSTM_PUB_sum_data is None:
            LSTM_PUB_sum_data = LSTM_PUB_df.copy()
        else:
            LSTM_PUB_sum_data = LSTM_PUB_sum_data.add(LSTM_PUB_df, fill_value=0)

    LSTM_PUB_mean_data = LSTM_PUB_sum_data.copy()
    LSTM_PUB_mean_data = LSTM_PUB_mean_data.divide(samples)
    LSTM_PUB_mean_data.index = test_basins_list
    LSTM_PUB_mean_data.columns = benchmark_list
    LSTM_PUB_mean_data.index.name = 'file_num'
    LSTM_PUB_mean_data = LSTM_PUB_mean_data[[benchmark]]
    LSTM_PUB_mean_data.index = test_basins_list  # Ensure index is set

    # Calculate 10% and 90% distributions
    LSTM_PUB_all_samples = np.array(LSTM_PUB_all_samples)  # Convert to numpy array for percentile calculations
    LSTM_PUB_10th_data = np.percentile(LSTM_PUB_all_samples, 10, axis=0)
    LSTM_PUB_90th_data = np.percentile(LSTM_PUB_all_samples, 90, axis=0)
    LSTM_PUB_10th_data = pd.DataFrame(LSTM_PUB_10th_data, columns=benchmark_list, index=test_basins_list)
    LSTM_PUB_90th_data = pd.DataFrame(LSTM_PUB_90th_data, columns=benchmark_list, index=test_basins_list)
    LSTM_PUB_10th_data = LSTM_PUB_10th_data[[benchmark]]
    LSTM_PUB_90th_data = LSTM_PUB_90th_data[[benchmark]]

    return LSTM_PUB_mean_data, LSTM_PUB_10th_data, LSTM_PUB_90th_data

def load_BC_data(benchmark):
    BC_data = f'hyper/out/{loc}/BC_700_0.001/results/BC_results_r700_s1_sr0.4_rr0.001_bma_eva.csv'
    BC_df = pd.read_csv(BC_data)
    BC_column = f'BC_r700_s1_sr0.4_rr0.001_bma_{benchmark}_eva'
    BC_data = BC_df[BC_column]

    return BC_data


def safe_min(sequence, default=float('inf')):
    return np.min(sequence) if len(sequence) > 0 else default

def safe_max(sequence, default=float('-inf')):
    return np.max(sequence) if len(sequence) > 0 else default

def safe_subtract(arr1, arr2):
    min_len = min(len(arr1), len(arr2))
    return arr1[:min_len] - arr2[:min_len]

output_dir = f'hyper/fig/{loc}/benchmark/Spatial_PUB_random_{reservoir_size}_{ridge_param}{nocal_tag}/'

os.makedirs(output_dir, exist_ok=True)
if BmaProx_cdf:
    output_dir_sub = os.path.join(output_dir, 'cdf_BmaProx')
else:
    if gauged_BC:
        output_dir_sub = os.path.join(output_dir, 'cdf_BCgauged')
        if LSTM_cdf:
            if BC_Prox_cdf:
                output_dir_sub = os.path.join(output_dir, 'cdf_woBmaProx_BCgauged_LSTM')
            else:
                output_dir_sub = os.path.join(output_dir, 'cdf_woBmaProx_woBcProx_BCgauged_LSTM')
        if BC_Prox_cdf:
            output_dir_sub = os.path.join(output_dir, 'cdf_woBmaProx_BCgauged')
    else:
        output_dir_sub = os.path.join(output_dir, 'cdf_woBmaProx')
os.makedirs(output_dir_sub, exist_ok=True)

calc_mode_to_name = {
    # 'BC_norm': 'BcProx',  # Removed BcProx from the calculation modes
    'PCA_norm': 'BcReg',
    'LSTM_norm': 'LSTM'
}

benchmark_limits_cdf = {
    'KGE': {'min': -0.2, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

benchmark_limits_cdf_summary = {
    'KGE': {'min': -1.5, 'max': 1},
    'NSE': {'min': -4.0, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -2.5, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

style_scheme = {
    'LSTM':     {'color': '#2ca02c', 'alpha': 0.1, 'linestyle': ':'},   # green
    'BcProx':   {'color': '#7b3294', 'alpha': 0.1, 'linestyle': '--'},  # Removed BcProx style
    'BcReg':    {'color': '#d7191c', 'alpha': 0.1, 'linestyle': '-'},   # red
    #'BmaProx':  {'color': '#1f78b4', 'alpha': 0.1, 'linestyle': '-.'}   # blue (if needed)
}


for benchmark in benchmark_list:
    plot_test_df_cdf = pd.DataFrame()  # at the end should include row: test basins, column train basins for the benchmark

    for calc_mode in calc_mode_to_name.keys():
        calc_name = calc_mode_to_name[calc_mode]

        output_dir_calc_mode = os.path.join(output_dir, calc_name)
        os.makedirs(output_dir_calc_mode, exist_ok=True)

        buf = f''
        
        global_y_min, global_y_max = float('inf'), float('-inf')

        for ce in cal_eva:
            for train_basin_int in train_basin_int_list:
                if train_int_tag:
                    train_int_tag_name = f'Train{train_basin_int}_'
                else:
                    train_int_tag_name = f''

                if loc == "JP" and train_basin_int == 70:
                    samples = 1
                else:
                    samples = samples_

                if 'BMA' in calc_mode:
                    BmaProx_column_test_data, BmaProx_10_data, BmaProx_90_data = load_BmaProx_data(train_basin_int, benchmark)

                if 'BC' in calc_mode and BC_Prox_cdf:
                    BcProx_column_test_data, BcProx_10_data, BcProx_90_data = load_BcProx_data(train_basin_int, benchmark)
                    plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                        f'{calc_name} {train_int_tag_name}{train_basin_int}': BcProx_column_test_data.values.flatten()
                    }, index=test_basins_list)], axis=1)

                elif calc_mode == 'PCA_norm':
                    for pc in range(1, n_components + 1):
                        BcReg_column_test_data, BcReg_10_data, BcReg_90_data = load_BcReg_data(pc, train_basin_int, benchmark)
                        if pc ==3:
                            plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                                f'{calc_name} {train_int_tag_name}{train_basin_int}': BcReg_column_test_data.values.flatten()
                            }, index=test_basins_list)], axis=1)

                elif calc_mode == 'LSTM_norm' and LSTM_cdf:  # Only process LSTM if LSTM_cdf is True
                    LSTM_PUB_column_test_data, LSTM_PUB_10_data, LSTM_PUB_90_data = load_LSTM_PUB_data(train_basin_int, benchmark)
                    if LSTM_PUB_column_test_data is not None:  # Ensure data exists
                        plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
                            f'{calc_name} {train_int_tag_name}{train_basin_int}': LSTM_PUB_column_test_data.values.flatten()
                        }, index=test_basins_list)], axis=1)

    # Sort plot_test_df_cdf to group PCA columns by PC1 to PC3 and sort the numbers before _PC from largest to smallest
    pca_columns = [col for col in plot_test_df_cdf.columns if 'PCA' in col]
    non_pca_columns = [col for col in plot_test_df_cdf.columns if 'PCA' not in col]
    pca_columns_sorted = sorted(pca_columns, key=lambda x: (x.split('_PC')[-1], -int(x.split('_PC')[0].split()[-1])))
    plot_test_df_cdf = plot_test_df_cdf[non_pca_columns + pca_columns_sorted]

    if gauged_BC:
        BC_data = load_BC_data(benchmark)
        plot_test_df_cdf = pd.concat([plot_test_df_cdf, pd.DataFrame({
            f'Gauged BC': BC_data
        }, index=test_basins_list)], axis=1)

    fig_cdf, ax_cdf = plt.subplots(nrows=1, ncols=1, figsize=(x_size_cdf, y_size_cdf))
    create_cdf_plot(ax_cdf, plot_test_df_cdf, benchmark,'test')
    # Set xlim and ylim for CDF plot
    if benchmark in benchmark_limits_cdf:
        ax_cdf.set_xlim(benchmark_limits_cdf[benchmark]['min'], benchmark_limits_cdf[benchmark]['max'])
    ax_cdf.set_ylim(0, 1)
    fig_cdf.tight_layout()

    if BmaProx_cdf:
        cdf_output_path = os.path.join(output_dir, f'cdf/{benchmark}_cdf{buf}_cal_eva.png')
    else:
        cdf_output_path = os.path.join(output_dir_sub, f'{benchmark}_cdf{buf}_cal_eva.png')
    fig_cdf.savefig(cdf_output_path)
    plt.close(fig_cdf)

    print(f'saved to {output_dir_calc_mode}')

    # --- Additional: CDF plot without LSTM ---
    plot_test_df_cdf_noLSTM = plot_test_df_cdf[[col for col in plot_test_df_cdf.columns if 'LSTM' not in col]]
    fig_cdf_noLSTM, ax_cdf_noLSTM = plt.subplots(nrows=1, ncols=1, figsize=(x_size_cdf, y_size_cdf))
    create_cdf_plot(ax_cdf_noLSTM, plot_test_df_cdf_noLSTM, benchmark, 'test')
    if benchmark in benchmark_limits_cdf:
        ax_cdf_noLSTM.set_xlim(benchmark_limits_cdf[benchmark]['min'], benchmark_limits_cdf[benchmark]['max'])
    ax_cdf_noLSTM.set_ylim(0, 1)
    fig_cdf_noLSTM.tight_layout()
    # Save with _noLSTM in filename
    if BmaProx_cdf:
        cdf_output_path_noLSTM = os.path.join(output_dir, f'cdf/{benchmark}_cdf{buf}_cal_eva_noLSTM.png')
    else:
        cdf_output_path_noLSTM = os.path.join(output_dir_sub, f'{benchmark}_cdf{buf}_cal_eva_noLSTM.png')
        
    fig_cdf_noLSTM.savefig(cdf_output_path_noLSTM)
    plt.close(fig_cdf_noLSTM)

    # Create summary plot where the x-axis is the number of training basins and the y-axis is the benchmark value
    fig_summary, ax_summary = plt.subplots(nrows=1, ncols=1, figsize=(x_size_sum, y_size_sum))
    for calc_mode in calc_mode_to_name.keys():
        calc_name = calc_mode_to_name[calc_mode]
        mean_values = []
        lower_bounds = []
        upper_bounds = []

        for train_basin_int in train_basin_int_list:
            if loc == "JP" and train_basin_int == 70:
                samples = 1
            else:
                samples = samples_

            if 'BMA' in calc_mode:
                BmaProx_column_test_data, BmaProx_10_data, BmaProx_90_data = load_BmaProx_data(train_basin_int, benchmark)
                mean_values.append(BmaProx_column_test_data.mean().values[0])
                lower_bounds.append(BmaProx_10_data.mean().values[0])
                upper_bounds.append(BmaProx_90_data.mean().values[0])
            elif 'BC' in calc_mode and BC_Prox_cdf:
                BcProx_column_test_data, BcProx_10_data, BcProx_90_data = load_BcProx_data(train_basin_int, benchmark)
                mean_values.append(BcProx_column_test_data.mean().values[0])
                lower_bounds.append(BcProx_10_data.mean().values[0])
                upper_bounds.append(BcProx_90_data.mean().values[0])
            elif 'PCA' in calc_mode:
                for pc in range(1, n_components + 1):
                    BcReg_column_test_data, BcReg_10_data, BcReg_90_data = load_BcReg_data(pc, train_basin_int, benchmark)
                    if pc == 3:
                        mean_values.append(BcReg_column_test_data.mean().values[0])
                        lower_bounds.append(BcReg_10_data.mean().values[0])
                        upper_bounds.append(BcReg_90_data.mean().values[0])
            elif 'LSTM' in calc_mode:
                LSTM_PUB_column_test_data, LSTM_PUB_10_data, LSTM_PUB_90_data = load_LSTM_PUB_data(train_basin_int, benchmark)
                mean_values.append(LSTM_PUB_column_test_data.mean().values[0])
                lower_bounds.append(LSTM_PUB_10_data.mean().values[0])
                upper_bounds.append(LSTM_PUB_90_data.mean().values[0])

        style = style_scheme.get(calc_name, {'color': 'black', 'alpha': 0.15, 'linestyle': '-'})

        # Optional: apply a tiny vertical jitter if needed
        jitter = 0.1 * list(style_scheme).index(calc_name)
        jittered_mean = [m + jitter for m in mean_values]
        jittered_lower = [l + jitter for l in lower_bounds]
        jittered_upper = [u + jitter for u in upper_bounds]

        # Plot mean with line style
        ax_summary.plot(train_basin_int_list, jittered_mean,
                        label=calc_name,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker='o',
                        linewidth=2,
                        markersize=5)

        # Plot shaded area with edge
        ax_summary.fill_between(train_basin_int_list, jittered_lower, jittered_upper,
                                color=style['color'],
                                alpha=style['alpha'],
                                edgecolor=style['color'],
                                linewidth=0.7)

        
    # Set xlim and ylim for summary plot
    ax_summary.set_xticks(train_basin_int_list)
    ax_summary.set_xticklabels(train_basin_int_list)
    ax_summary.set_ylim(benchmark_limits_cdf_summary[benchmark]['min'], benchmark_limits_cdf_summary[benchmark]['max'])
    ax_summary.set_title(f'{benchmark} and No. Training Basin\n', fontsize=fs)
    ax_summary.set_xlabel(f'No. of Training Basins', fontsize=fs_tick)
    ax_summary.set_ylabel(f'{benchmark}', fontsize=fs_tick)
    ax_summary.xaxis.set_tick_params(labelsize=fs_tick)
    ax_summary.yaxis.set_tick_params(labelsize=fs_tick) 
    ax_summary.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs * 1.3)
    fig_summary.tight_layout()

    ax_summary.grid(True, linestyle='--', alpha=0.4)
    output_summary_path = os.path.join(output_dir, f'summary')
    os.makedirs(output_summary_path, exist_ok=True)
    plt.savefig(os.path.join(output_summary_path, f'{benchmark}_summary{buf}_cal_eva.png'))

    # --- Additional: summary plot without LSTM ---
    fig_summary_noLSTM, ax_summary_noLSTM = plt.subplots(nrows=1, ncols=1, figsize=(x_size_sum, y_size_sum))
    for calc_mode in calc_mode_to_name.keys():
        calc_name = calc_mode_to_name[calc_mode]
        if calc_name == 'LSTM':
            continue  # skip LSTM
        mean_values = []
        lower_bounds = []
        upper_bounds = []
        for train_basin_int in train_basin_int_list:
            if loc == "JP" and train_basin_int == 70:
                samples = 1
            else:
                samples = samples_
            if 'BMA' in calc_mode:
                BmaProx_column_test_data, BmaProx_10_data, BmaProx_90_data = load_BmaProx_data(train_basin_int, benchmark)
                mean_values.append(BmaProx_column_test_data.mean().values[0])
                lower_bounds.append(BmaProx_10_data.mean().values[0])
                upper_bounds.append(BmaProx_90_data.mean().values[0])
            elif 'BC' in calc_mode and BC_Prox_cdf:
                BcProx_column_test_data, BcProx_10_data, BcProx_90_data = load_BcProx_data(train_basin_int, benchmark)
                mean_values.append(BcProx_column_test_data.mean().values[0])
                lower_bounds.append(BcProx_10_data.mean().values[0])
                upper_bounds.append(BcProx_90_data.mean().values[0])
            elif 'PCA' in calc_mode:
                for pc in range(1, n_components + 1):
                    BcReg_column_test_data, BcReg_10_data, BcReg_90_data = load_BcReg_data(pc, train_basin_int, benchmark)
                    if pc == 3:
                        mean_values.append(BcReg_column_test_data.mean().values[0])
                        lower_bounds.append(BcReg_10_data.mean().values[0])
                        upper_bounds.append(BcReg_90_data.mean().values[0])
        style = style_scheme.get(calc_name, {'color': 'black', 'alpha': 0.15, 'linestyle': '-'})
        jitter = 0.1 * list(style_scheme).index(calc_name)
        jittered_mean = [m + jitter for m in mean_values]
        jittered_lower = [l + jitter for l in lower_bounds]
        jittered_upper = [u + jitter for u in upper_bounds]
        ax_summary_noLSTM.plot(train_basin_int_list, jittered_mean,
                               label=calc_name,
                               color=style['color'],
                               linestyle=style['linestyle'],
                               marker='o',
                               linewidth=2,
                               markersize=5)
        ax_summary_noLSTM.fill_between(train_basin_int_list, jittered_lower, jittered_upper,
                                      color=style['color'],
                                      alpha=style['alpha'],
                                      edgecolor=style['color'],
                                      linewidth=0.7)
    ax_summary_noLSTM.set_xticks(train_basin_int_list)
    ax_summary_noLSTM.set_xticklabels(train_basin_int_list)
    ax_summary_noLSTM.set_ylim(benchmark_limits_cdf_summary[benchmark]['min'], benchmark_limits_cdf_summary[benchmark]['max'])
    ax_summary_noLSTM.set_title(f'{benchmark} and No. of Training Basin\n', fontsize=fs)
    ax_summary_noLSTM.set_xlabel(f'No. of Training Basins', fontsize=fs_tick)
    ax_summary_noLSTM.set_ylabel(f'{benchmark}', fontsize=fs_tick)
    ax_summary_noLSTM.xaxis.set_tick_params(labelsize=fs_tick)
    ax_summary_noLSTM.yaxis.set_tick_params(labelsize=fs_tick)
    ax_summary_noLSTM.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs * 1.0)
    fig_summary_noLSTM.tight_layout()
    ax_summary_noLSTM.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(output_summary_path, f'{benchmark}_summary{buf}_cal_eva_noLSTM.png'))

    if LSTM_cdf:
        fig_summary_LSTM, ax_summary_LSTM = plt.subplots(nrows=1, ncols=1, figsize=(x_size_sum, y_size_sum))
        mean_values = []
        lower_bounds = []
        upper_bounds = []

        for train_basin_int in train_basin_int_list:
            if loc == "JP" and train_basin_int == 70:
                samples = 1
            else:
                samples = samples_

            LSTM_PUB_column_test_data, LSTM_PUB_10_data, LSTM_PUB_90_data = load_LSTM_PUB_data(train_basin_int, benchmark)
            if LSTM_PUB_column_test_data is not None:  # Ensure data exists
                mean_values.append(LSTM_PUB_column_test_data.mean().values[0])
                lower_bounds.append(LSTM_PUB_10_data.mean().values[0])
                upper_bounds.append(LSTM_PUB_90_data.mean().values[0])
            else:  # Handle case where LSTM_PUB_column_test_data is None
                mean_values.append(None)
                lower_bounds.append(None)
                upper_bounds.append(None)

        # Filter out None values before plotting
        valid_indices = [i for i, value in enumerate(mean_values) if value is not None]
        mean_values = [mean_values[i] for i in valid_indices]
        lower_bounds = [lower_bounds[i] for i in valid_indices]
        upper_bounds = [upper_bounds[i] for i in valid_indices]
        train_basin_int_list = [train_basin_int_list[i] for i in valid_indices]

        style = style_scheme.get('LSTM', {'color': 'black', 'alpha': 0.15, 'linestyle': '-'})

        # Plot mean with line style
        ax_summary_LSTM.plot(train_basin_int_list, mean_values,
                             label='LSTM',
                             color=style['color'],
                             linestyle=style['linestyle'],
                             marker='o',
                             linewidth=2,
                             markersize=5)

        # Plot shaded area with edge
        ax_summary_LSTM.fill_between(train_basin_int_list, lower_bounds, upper_bounds,
                                     color=style['color'],
                                     alpha=style['alpha'],
                                     edgecolor=style['color'],
                                     linewidth=0.7)

        # Set xlim and ylim for summary plot
        ax_summary_LSTM.set_xticks(train_basin_int_list)
        ax_summary_LSTM.set_xticklabels(train_basin_int_list)
        ax_summary_LSTM.set_ylim(benchmark_limits_cdf_summary[benchmark]['min'], benchmark_limits_cdf_summary[benchmark]['max'])
        ax_summary_LSTM.set_title(f'{benchmark} and No. of Training Basin (LSTM Only)\n', fontsize=fs)
        ax_summary_LSTM.set_xlabel(f'No. of Training Basins', fontsize=fs_tick)
        ax_summary_LSTM.set_ylabel(f'{benchmark}', fontsize=fs_tick)
        ax_summary_LSTM.xaxis.set_tick_params(labelsize=fs_tick)
        ax_summary_LSTM.yaxis.set_tick_params(labelsize=fs_tick)
        ax_summary_LSTM.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs * 1.0)
        fig_summary_LSTM.tight_layout()
        ax_summary_LSTM.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(output_summary_path, f'{benchmark}_summary{buf}_cal_eva_LSTM.png'))

    # --- Additional: summary plot for LSTM and BC Reg only ---
    fig_summary_LSTM_BCReg, ax_summary_LSTM_BCReg = plt.subplots(nrows=1, ncols=1, figsize=(x_size_sum, y_size_sum))
    mean_values_dict = {'LSTM': [], 'BcReg': []}
    lower_bounds_dict = {'LSTM': [], 'BcReg': []}
    upper_bounds_dict = {'LSTM': [], 'BcReg': []}

    for calc_mode in ['LSTM_norm', 'PCA_norm']:
        calc_name = calc_mode_to_name[calc_mode]
        for train_basin_int in train_basin_int_list:
            if loc == "JP" and train_basin_int == 70:
                samples = 1
            else:
                samples = samples_

            if calc_mode == 'LSTM_norm' and LSTM_cdf:
                LSTM_PUB_column_test_data, LSTM_PUB_10_data, LSTM_PUB_90_data = load_LSTM_PUB_data(train_basin_int, benchmark)
                mean_values_dict['LSTM'].append(LSTM_PUB_column_test_data.mean().values[0])
                lower_bounds_dict['LSTM'].append(LSTM_PUB_10_data.mean().values[0])
                upper_bounds_dict['LSTM'].append(LSTM_PUB_90_data.mean().values[0])
            elif calc_mode == 'PCA_norm':
                for pc in range(1, n_components + 1):
                    BcReg_column_test_data, BcReg_10_data, BcReg_90_data = load_BcReg_data(pc, train_basin_int, benchmark)
                    if pc == 3:
                        mean_values_dict['BcReg'].append(BcReg_column_test_data.mean().values[0])
                        lower_bounds_dict['BcReg'].append(BcReg_10_data.mean().values[0])
                        upper_bounds_dict['BcReg'].append(BcReg_90_data.mean().values[0])

    for calc_name in ['LSTM', 'BcReg']:
        style = style_scheme.get(calc_name, {'color': 'black', 'alpha': 0.15, 'linestyle': '-'})
        ax_summary_LSTM_BCReg.plot(train_basin_int_list, mean_values_dict[calc_name],
                                   label=calc_name,
                                   color=style['color'],
                                   linestyle=style['linestyle'],
                                   marker='o',
                                   linewidth=2,
                                   markersize=5)
        ax_summary_LSTM_BCReg.fill_between(train_basin_int_list, lower_bounds_dict[calc_name], upper_bounds_dict[calc_name],
                                           color=style['color'],
                                           alpha=style['alpha'],
                                           edgecolor=style['color'],
                                           linewidth=0.7)

    # Set xlim and ylim for summary plot
    ax_summary_LSTM_BCReg.set_xticks(train_basin_int_list)
    ax_summary_LSTM_BCReg.set_xticklabels(train_basin_int_list)
    ax_summary_LSTM_BCReg.set_ylim(benchmark_limits_cdf_summary[benchmark]['min'], benchmark_limits_cdf_summary[benchmark]['max'])
    ax_summary_LSTM_BCReg.set_title(f'{benchmark} and No. of Training Basin (LSTM and BC Reg Only)\n', fontsize=fs)
    ax_summary_LSTM_BCReg.set_xlabel(f'No. of Training Basins', fontsize=fs_tick)
    ax_summary_LSTM_BCReg.set_ylabel(f'{benchmark}', fontsize=fs_tick)
    ax_summary_LSTM_BCReg.xaxis.set_tick_params(labelsize=fs_tick)
    ax_summary_LSTM_BCReg.yaxis.set_tick_params(labelsize=fs_tick)
    ax_summary_LSTM_BCReg.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs * 1.0)
    fig_summary_LSTM_BCReg.tight_layout()
    ax_summary_LSTM_BCReg.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(output_summary_path, f'{benchmark}_summary{buf}_cal_eva_LSTM_BCReg.png'))

print("DONE!")
