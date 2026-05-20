"""
This code is associated to the following paper:

Funato, M., Sawada, Y., "Multi-Model Ensemble and Reservoir Computing for River Discharge Prediction in Ungauged Basins".
currently under submission
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

n_components = 3
ver = 2
loc = "JP"
fs = 22

test_y_ticks = True
LSTM_data = True

ver_name = "ver1_1" if ver == 1 else "ver2_0"

samples = 100

ridge_param = 1.0

if loc == "JP":
    file_tot_num = 87
    reservoir_size = 200
    train_basin_int = 15
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
    region_list = ['Hokkaido', 'Tohoku', 'North Central', 'South Central', 'West']
    region_list_df = pd.read_csv(f'data/river_basin/dataset_{loc}/pub_region_list_{ver_name}.csv')


benchmark_list = ["KGE", "NSE", "E1", "VE", "d", "RMSE", "MAE"]

benchmark_limits = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -2, 'max': 1},
    'E1': {'min': -2, 'max': 1},    
    'VE': {'min': -2, 'max': 1},
    'd': {'min': 0, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

def load_BcProx_data(train_basin_int, benchmark):
    BcProx_sum_data = None
    for sample in range(1, samples + 1):
        BcProx_data = f'out/{loc}/BcProx/random/{reservoir_size}_{ridge_param}/Train{train_basin_int}/test_basin/results/BcProx_results_Train{train_basin_int}_sample{sample}_eva.csv'
        BcProx_df = pd.read_csv(BcProx_data, index_col=0)
        BcProx_df = BcProx_df.sort_values(by='file_num')
        BcProx_df = pd.DataFrame(BcProx_df.values)

        if BcProx_sum_data is None:
            BcProx_sum_data = BcProx_df.copy()
        else:
            BcProx_sum_data = BcProx_sum_data.add(BcProx_df, fill_value=0)

    
    BcProx_mean_data = BcProx_sum_data.copy()
    BcProx_mean_data = BcProx_mean_data.divide(samples)

    BcProx_mean_data.columns = benchmark_list

    BcProx_mean_data.index.name = 'file_num'

    BcProx_mean_data = BcProx_mean_data[[benchmark]]
    BcProx_mean_data.index = test_basins_list  # Ensure index is set


    return BcProx_mean_data
    
def load_BcReg_data(pc, train_basin_int, benchmark):
    BcReg_sum_data = None
    for sample in range(1, samples + 1):
        BcReg_data = f'out/{loc}/BcReg/random/{reservoir_size}_{ridge_param}/Train{train_basin_int}/test_basin/results/sample{sample}/BcReg_results_Train{train_basin_int}_sample{sample}_rev_PC{pc}_eva.csv'
        BcReg_df = pd.read_csv(BcReg_data, index_col=0)
        BcReg_df = BcReg_df.sort_values(by='file_num')
        BcReg_df = pd.DataFrame(BcReg_df.values)

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

    return BcReg_mean_data

def load_LSTM_PUB_data(train_basin_int, benchmark):
    LSTM_PUB_sum_data = None
    LSTM_PUB_all_samples = []  # Store all samples for percentile calculations
    for sample in range(1, samples + 1):
        LSTM_PUB_data = f'out/{loc}/LSTM_PUB/random/ensemble/Train{train_basin_int}/test_basin/results/LSTM_PUB_results_Train{train_basin_int}_sample{sample}_eva.csv'
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

    return LSTM_PUB_mean_data


def load_BcProx_region_data(region, benchmark, train_test):
    BcProx_region_data = pd.read_csv(f'out/{loc}/BcProx/region/{reservoir_size}_{ridge_param}/{region}/{train_test}_basin/results/BcProx_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_eva.csv')
    BcProx_region_data = BcProx_region_data.sort_values(by='file_num')
    BcProx_region_data = BcProx_region_data[BcProx_region_data['file_num'].isin(test_basins_list)]
    BcProx_region_data = BcProx_region_data[f'BcProx_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
    return BcProx_region_data

def load_BcReg_region_data(pc, region, benchmark, train_test):
    BcReg_region_data = pd.read_csv(f'out/{loc}/BcReg/region/{reservoir_size}_{ridge_param}/{region}/{train_test}_basin/results/BcReg_results_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_rev_PC{pc}_eva.csv')
    BcReg_region_data = BcReg_region_data.sort_values(by='file_num')
    BcReg_region_data = BcReg_region_data[BcReg_region_data['file_num'].isin(test_basins_list)]
    BcReg_region_data = BcReg_region_data[f'BcReg_r{reservoir_size}_s1_sr0.4_rr{ridge_param}_{benchmark}_eva']
    return BcReg_region_data

def load_LSTM_PUB_region_data(region, benchmark, train_test):
    LSTM_PUB_region_data = pd.read_csv(f'out/{loc}/LSTM_PUB/region/ensemble/{region}/{train_test}_basin/results/LSTM_PUB_results_mean_eva.csv')
    LSTM_PUB_region_data = LSTM_PUB_region_data.sort_values(by='file_num')
    LSTM_PUB_region_data = LSTM_PUB_region_data[LSTM_PUB_region_data['file_num'].isin(test_basins_list)]
    LSTM_PUB_region_data = LSTM_PUB_region_data[f'LSTM_PUB_{benchmark}_eva']
    return LSTM_PUB_region_data

for pc_n in range(1, n_components+1):
    output_dir = f'fig/{loc}/benchmark/Spatial_PUB_regional_{reservoir_size}_{ridge_param}/box/PC{pc_n}'
    if LSTM_data:
        output_dir = f'fig/{loc}/benchmark/Spatial_PUB_regional_{reservoir_size}_{ridge_param}/box_LSTM/PC{pc_n}'
    os.makedirs(output_dir, exist_ok=True)
    for benchmark in benchmark_list:
        plot_df_train = pd.DataFrame()
        plot_df_test = pd.DataFrame()
        plot_df_train_noDist = pd.DataFrame()
        plot_df_test_noDist = pd.DataFrame()

        train_num = []
        test_num = []
        tick_label_train = []
        tick_label_test = []
        color_list_train = []
        color_list_test = []

        # add BCPUB data for distributed
        BcProx_data = load_BcProx_data(train_basin_int, benchmark)
        plot_df_test = pd.concat([plot_df_test, BcProx_data], axis=1)
        # For noDist, add a column of NaN with the same index and same shape as BcProx_data for spacing
        plot_df_test_noDist = pd.concat([plot_df_test_noDist, pd.DataFrame([float('nan')] * len(BcProx_data), index=BcProx_data.index)], axis=1)
        tick_label_test.append(f'Random')
        test_num.append(train_basin_int)
        color_list_test.append('#9982ab')
        
        train_num_list = [17,14,13,13,13]

        # add BCPUB data for each region
        for region in region_list:
            BcProx_region_data_train = load_BcProx_region_data(region, benchmark, 'train')
            plot_df_train = pd.concat([plot_df_train, BcProx_region_data_train], axis=1)
            tick_label_train.append(f'{region}')
            train_num.append(train_num_list[region_list.index(region)])
            color_list_train.append('#D0B8E6')

            BcProx_region_data_test = load_BcProx_region_data(region, benchmark, 'test')
            plot_df_test = pd.concat([plot_df_test, BcProx_region_data_test], axis=1)
            plot_df_test_noDist = pd.concat([plot_df_test_noDist, BcProx_region_data_test], axis=1)
            tick_label_test.append(f'{region}')
            test_num.append(train_num_list[region_list.index(region)])
            color_list_test.append('#D0B8E6')

        # add BCPCA data for distributed
        BcReg_data = load_BcReg_data(pc_n, train_basin_int, benchmark)
        plot_df_test = pd.concat([plot_df_test, BcReg_data], axis=1)
        plot_df_test_noDist = pd.concat([plot_df_test_noDist, pd.DataFrame([float('nan')] * len(BcReg_data), index=BcReg_data.index)], axis=1)
        tick_label_test.append(f'Random')
        test_num.append(train_basin_int)
        color_list_test.append('#C7625E')
        # add BCPCA data for each region

        for region in region_list:
            BcReg_region_data_train = load_BcReg_region_data(pc_n, region, benchmark, 'train')
            plot_df_train = pd.concat([plot_df_train, BcReg_region_data_train], axis=1)
            tick_label_train.append(f'{region}')
            train_num.append(train_num_list[region_list.index(region)])
            color_list_train.append('#E8B4A9')
            BcReg_region_data_test = load_BcReg_region_data(pc_n, region, benchmark, 'test')
            plot_df_test = pd.concat([plot_df_test, BcReg_region_data_test], axis=1)
            plot_df_test_noDist = pd.concat([plot_df_test_noDist, BcReg_region_data_test], axis=1)
            tick_label_test.append(f'{region}')
            test_num.append(train_num_list[region_list.index(region)])
            color_list_test.append('#E8B4A9')

        # add LSTM_PUB distributed data (test only)
        if LSTM_data:
            LSTM_PUB_data = load_LSTM_PUB_data(train_basin_int, benchmark)
            plot_df_test = pd.concat([plot_df_test, LSTM_PUB_data], axis=1)
            plot_df_test_noDist = pd.concat([plot_df_test_noDist, pd.DataFrame([float('nan')] * len(LSTM_PUB_data), index=LSTM_PUB_data.index)], axis=1)
            tick_label_test.append(f'Random')
            test_num.append(train_basin_int)
            color_list_test.append('#7FB77E')  # Use a distinct color for LSTM distributed
            for region in region_list:
                LSTM_PUB_region_data_test = load_LSTM_PUB_region_data(region, benchmark, 'test')
                plot_df_test = pd.concat([plot_df_test, LSTM_PUB_region_data_test], axis=1)
                plot_df_test_noDist = pd.concat([plot_df_test_noDist, LSTM_PUB_region_data_test], axis=1)
                tick_label_test.append(f'{region} ')
                test_num.append(train_num_list[region_list.index(region)])
                color_list_test.append('#c8e0c8')  # Use a distinct color for LSTM
                #test_num.append()

        plot_df_train_noDist = plot_df_train.copy()

        ### PLOT
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))

        # Box plot for calibration data
        plot_df_train.columns = tick_label_train
        boxplot_train = plot_df_train.boxplot(ax=axes[0], patch_artist=True, return_type='dict')
        axes[0].set_title(f'Training Basins\n', fontsize=fs)
        #axes[0].set_xlabel('Region', fontsize=fs)
        axes[0].set_xticklabels([f'{label} ({file_num})' for label, file_num in zip(tick_label_train, train_num)], rotation=90, fontsize=fs)
        axes[0].set_ylabel(f'{benchmark}', fontsize=fs)
        axes[0].tick_params(axis='y', labelsize=fs)
        
        # Box plot for evaluation data
        plot_df_test.columns = tick_label_test
        boxplot_test = plot_df_test.boxplot(ax=axes[1], patch_artist=True, return_type='dict')
        axes[1].set_title(f'Test Basins\n', fontsize = fs)
        #axes[1].set_xlabel('Region', fontsize=fs)
        axes[1].set_xticklabels([f'{label} ({file_num})' for label, file_num in zip(tick_label_test, test_num)], rotation=90, fontsize=fs)
        if test_y_ticks:
            axes[1].set_ylabel(f'{benchmark}' , fontsize=fs)
            axes[1].tick_params(axis='y', labelsize=fs)
        else:
            axes[1].tick_params(axis='y', labelsize=0)

        if test_y_ticks:
            plt.subplots_adjust(bottom=0.39, wspace=0.25)  # Increase bottom margin and space between columns
        else:
            plt.subplots_adjust(bottom=0.4, wspace=0.07)  # Increase bottom margin and space between columns

        axes[0].tick_params(axis='x', labelsize=fs, pad=10)  # Adjust label size and padding
        axes[1].tick_params(axis='x', labelsize=fs, pad=10)

        axes[0].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
        axes[1].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

        # Ensure color_list dynamically matches the number of boxes
        for patch, color in zip(boxplot_train['boxes'], color_list_train):
            patch.set_facecolor(color)

        for patch, color in zip(boxplot_test['boxes'], color_list_test):
            patch.set_facecolor(color)

        # Change the median line color to black
        for median in boxplot_train['medians']:
            median.set_color('black')
        for median in boxplot_test['medians']:
            median.set_color('black')

        # Add horizontal brackets
        def add_bracket(ax, text, start, end, height, y_offset):
            ax.annotate('', xy=(start, height), xytext=(end, height), 
                        arrowprops={'arrowstyle': '-', 'color': 'black', 'linewidth': 1.5})
            ax.text((start + end) * 0.5, height + y_offset, text, ha='center', va='bottom', fontsize=fs)

        if benchmark == 'KGE' or benchmark == 'NSE':
            height = -2.6
        elif benchmark == 'E1':
            height = -4.3
        elif benchmark == 'RMSE':
            height = -7.8


        #plt.tight_layout()
        plt.savefig(f'{output_dir}/{benchmark}_box_regions_PC{pc_n}.jpg')
        plt.close()

        # --------- NEW: Plot noDist version ---------
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))

        # Replace 'Random' with '' in tick_label_test for noDist plot
        tick_label_test_noDist = [label if 'Random' not in label else '' for label in tick_label_test]

        # Box plot for calibration data (noDist)
        plot_df_train_noDist.columns = tick_label_train
        boxplot_train_noDist = plot_df_train_noDist.boxplot(ax=axes[0], patch_artist=True, return_type='dict')
        axes[0].set_title(f'Training Basins\n', fontsize=fs)
        axes[0].set_xticklabels([f'{label} ({file_num})' for label, file_num in zip(tick_label_train, train_num)], rotation=90, fontsize=fs)
        axes[0].set_ylabel(f'{benchmark}', fontsize=fs)
        axes[0].tick_params(axis='y', labelsize=fs)

        # Box plot for evaluation data (noDist)
        plot_df_test_noDist.columns = tick_label_test_noDist
        boxplot_test_noDist = plot_df_test_noDist.boxplot(ax=axes[1], patch_artist=True, return_type='dict')
        axes[1].set_title(f'Test Basins\n', fontsize = fs)
        axes[1].set_xticklabels([f'{label} ({file_num})' for label, file_num in zip(tick_label_test_noDist, test_num)], rotation=90, fontsize=fs)
        if test_y_ticks:
            axes[1].set_ylabel(f'{benchmark}' , fontsize=fs)
            axes[1].tick_params(axis='y', labelsize=fs)
        else:
            axes[1].tick_params(axis='y', labelsize=0)

        if test_y_ticks:
            plt.subplots_adjust(bottom=0.39, wspace=0.25)
        else:
            plt.subplots_adjust(bottom=0.4, wspace=0.07)

        axes[0].tick_params(axis='x', labelsize=fs, pad=10)
        axes[1].tick_params(axis='x', labelsize=fs, pad=10)

        axes[0].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])
        axes[1].set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

        # Use same color logic as above
        for patch, color in zip(boxplot_train_noDist['boxes'], color_list_train):
            patch.set_facecolor(color)
        for patch, color in zip(boxplot_test_noDist['boxes'], color_list_test):
            patch.set_facecolor(color)
        for median in boxplot_train_noDist['medians']:
            median.set_color('black')
        for median in boxplot_test_noDist['medians']:
            median.set_color('black')

        plt.savefig(f'{output_dir}/{benchmark}_box_regions_PC{pc_n}_noRandom.jpg')
        plt.close()
        # --------- END NEW ---------
