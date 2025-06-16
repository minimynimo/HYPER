# Creates box plots, line plots, and point plots for each benchmark type (KGE, NSE, logNSE, E1, Erel, VE, d, RMSE, MAE) for both calibration and evaluation periods.
# use for BMA, RC, RCH-bma, RCH-bc, and bias correction
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

########
data_type = 'BMA & RC & RCH & BC'
#data_type = 'BMA & RC & BC'
#data_type = 'RCH & BC'
#data_type = 'RC & RCH & BC'

loc = "JP"

LSTM = False

simple_colors = True
#simple_colors = False

#RC
#reservoir_size = 200
reservoir_size = 700
ridge_param = 0.001 
#ridge_param = 1.0

##LSTM
num_epochs = 200
hidden_size = 20
learning_rate = 1e-3
window_size = 365
batch_size = 512
dropout_rate = 0.1

fs = 20 
#########
file_tag = f'r{reservoir_size}_sr0.4_rr{ridge_param}'
file_tag_LSTM = f'h{hidden_size}_lr{learning_rate}_e{num_epochs}_w{window_size}_b{batch_size}_d{dropout_rate}'

if LSTM:
    output_dir = f'hyper/fig/{loc}/benchmark/{data_type}_wLSTM_{reservoir_size}_{ridge_param}'
else:
    output_dir = f'hyper/fig/{loc}/benchmark/{data_type}_{reservoir_size}_{ridge_param}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'box'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'point'), exist_ok=True)

model_list = [ "m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08", "m09", "m10",
              "m11", "m12", "m13", "m14", "m15", "m16", "m17", "m18", "m19", "m20",
              "m21", "m22", "m23", "m24", "m25", "m26", "m27", "m28", "m29", "m30",
              "m31", "m32", "m33", "m34", "m35", "m36", "m37", "m38", "m39",
              "m42", "m43", "m44", "m46"]

benchmark_limit_avebma = {
    'KGE': {'min': -5, 'max': 1},
    'NSE': {'min': -5, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -5, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max':10},
    'MAE': {'min': 0, 'max': 10}
}

benchmark_limits = {
    'KGE': {'min': -1, 'max': 1},
    'NSE': {'min': -1, 'max': 1},
    'logNSE': {'min': -150, 'max': 1},
    'E1': {'min': -1, 'max': 1},
    'VE': {'min': -2, 'max': 1},
    'd': {'min': -2, 'max': 1},
    'RMSE': {'min': 0, 'max': 10},
    'MAE': {'min': 0, 'max': 10}
}

ce_names = ['Calibration', 'Evaluation']

def create_box_plot(ax, plot_df_cal, plot_df_eva, benchmark, ce):
    plot_dfs = [plot_df_cal, plot_df_eva]
    
    for i, plot_df in enumerate(plot_dfs):
        ce_name = ce_names[i]
        
        # Define the special columns and filter out any that are missing from plot_df
        special_columns_sub = ['BMA', 'RC', 'RCH', 'BC']
        if LSTM:
            special_columns_sub.append('LSTM')
        
        # Default assignment for special_columns
        special_columns = []

        # Refactor redundant if-else statements for data_type and special_columns
        special_columns_map = {
            'BMA & RC & RCH & BC': ['BMA', 'RC', 'RCH', 'BC'],
            'BMA & RC & BC': ['BMA', 'RC', 'BC'],
            'RCH & BC': ['RCH', 'BC'],
            'RC & RCH & BC': ['RC', 'RCH', 'BC'],
        }

        special_columns = special_columns_map.get(data_type, [])
        if LSTM:
            special_columns.append('LSTM')

        # Filter special_columns to include only those present in plot_df
        special_columns = [col for col in special_columns if col in plot_df.columns]

        # Separate the remaining columns
        remaining_columns = [col for col in plot_df.columns if col not in special_columns_sub]
        
        # Create boxplot for special columns
        boxplot = ax[i].boxplot([plot_df[col].values for col in special_columns],
                                patch_artist=True, positions=range(1, len(special_columns) + 1),
                                widths=0.6, labels=special_columns)
        
        # Create boxplot for remaining columns with no gap between them (tight positions)
        remaining_positions = [len(special_columns)+ 1 + j * 0.1 for j in range(len(remaining_columns))]
        remaining_boxplot = ax[i].boxplot([plot_df[col].values for col in remaining_columns],
                                          patch_artist=True, positions=remaining_positions,
                                          widths=0.1)  # Small width to make the boxes appear tight
        
        # Set title and labels
        ax[i].set_title(f'{benchmark} - {ce_name}\n', fontsize = fs*1.5)
        ax[i].set_title(f'{ce_name}', fontsize = fs)
        ax[i].set_ylabel(benchmark, fontsize = fs)

        # Set xticks with the remaining columns compressed into a single tick label
        middle_remaining_position = (remaining_positions[0] + remaining_positions[-1]) / 2
        
        # Set xticks with the remaining columns compressed into a single tick label
        ticks = list(range(1, len(special_columns) + 1)) + [middle_remaining_position]    
        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(special_columns + [f'{str(len(model_list))} MARRMoT models'], fontsize  = fs)
        ax[i].set_yticklabels(ax[i].get_yticks(), fontsize = fs)

        ax[i].grid(axis = 'y', linewidth = 0.4)
        
        # Define the box colors for the special columns
        # Refactor redundant if-else statements to define box_colors
        if data_type in ['BMA & RC & RCH & BC', 'RCH & BC', 'RC & RCH & BC', 'BMA & RC & BC', 'AVE & BMA & RC & RCH & BC & RCBC-BMA']:
            box_colors = get_box_colors(special_columns, simple_colors)
        else:
            box_colors = ['gainsboro' for _ in special_columns]

        # Set color for the special columns
        for patch, color in zip(boxplot['boxes'], box_colors):
            patch.set_facecolor(color)
        
        # Set all remaining columns' boxes to gray
        for patch in remaining_boxplot['boxes']:
            patch.set_facecolor('gainsboro')
        
        # Set median line properties for all boxes
        for median in boxplot['medians'] + remaining_boxplot['medians']:
            median.set_color('black')  # Set the desired color
            median.set_linewidth(2)         # Optionally, adjust the thickness

        ax[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

def create_point_plot(ax, plot_df_cal, plot_df_eva, benchmark, ce):
    plot_dfs = [plot_df_cal, plot_df_eva]
    
    for i, plot_df in enumerate(plot_dfs):
        ce_name = ce_names[i]
        positions = list(range(1, len(plot_df.columns) + 1))
        for j in range(len(plot_df)):
            ax[i].plot(positions, plot_df.iloc[j], marker='.', linewidth=0, color='royalblue')
        ax[i].set_title(f'{benchmark} - {ce_name}\n', fontsize = fs)
        ax[i].set_ylabel(benchmark)
        ax[i].set_xlabel('Model Type')
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(plot_df.columns, rotation=90)

        ax[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Define dictionaries for box colors
color_dict_simple = {
    'BMA': '#f2e8dc',
    'RC': '#f2e8dc',
    'RCH': '#FFC857',
    'BC': '#F45B69',
    'LSTM': '#7cc7bc'
}

color_dict_detailed = {
    'BMA': '#a6bddb',
    'RC': '#fdae6b',
    'RCH': '#66c2a4',
    'BC': '#fb8072',
    'LSTM': 'pink'
}

def get_box_colors(columns, simple_colors):
    if simple_colors:
        return [color_dict_simple.get(col, 'gainsboro') for col in columns]
    else:
        return [color_dict_detailed.get(col, 'gainsboro') for col in columns]

#buf = "_ave-bma-indivmodel"
buf = ""

benchmark_list = ["KGE","NSE","E1","VE", "d","RMSE","MAE"]

for benchmark in benchmark_list:
    for ce in ['cal', 'eva']:
        # Create a figure with 2 subplots: one for cal and one for eva
        fig_box, axes_box = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        fig_point, axes_point = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        global_y_min, global_y_max = float('inf'), float('-inf')


        BMA_file_path = f'hyper/out/{loc}/BMA/results/BMA_results_{ce}.csv'
        if LSTM:
            LSTM_file_path = f'hyper/out/{loc}/LSTM/results/LSTM_results_{file_tag_LSTM}_{ce}_test.csv'
        if data_type == 'BMA & RC & RCH & BC' or data_type == 'RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'BMA & RC & BC':
            RC_file_path = f'hyper/out/{loc}/RC_{reservoir_size}_{ridge_param}/results/RC_results_{file_tag}_{ce}.csv'
            RCH_file_path = f'hyper/out/{loc}/RCH_{reservoir_size}_{ridge_param}/results/RCH_results_{file_tag}_{ce}.csv'
            HYPER_file_path = f'hyper/out/{loc}/HYPER_{reservoir_size}_{ridge_param}/results/HYPER_results_{file_tag}_{ce}.csv'
        if data_type == 'RCH & BC':
            RCH_file_path = f'hyper/out/{loc}/RCH_{reservoir_size}_{ridge_param}/results/RCH_results_{file_tag}_{ce}.csv'
            HYPER_file_path = f'hyper/out/{loc}/HYPER_{reservoir_size}_{ridge_param}/results/HYPER_results_{file_tag}_{ce}.csv'

        BMA_df = pd.read_csv(BMA_file_path)
        if LSTM:
            LSTM_df = pd.read_csv(LSTM_file_path) 
        if data_type == 'BMA & RC & RCH & BC' or data_type == 'RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'BMA & RC & BC':
            RC_df = pd.read_csv(RC_file_path)
            RCH_df = pd.read_csv(RCH_file_path)
            HYPER_df = pd.read_csv(HYPER_file_path)

        BMA_column = f'BMA_{benchmark}_{ce}'
        LSTM_column = f'LSTM_{file_tag_LSTM}_{benchmark}_{ce}'
        RC_column = f'RC_{file_tag}_{benchmark}_{ce}'
        RCH_column = f'RCH_{file_tag}_{benchmark}_{ce}'
        HYPER_column = f'HYPER_{file_tag}_{benchmark}_{ce}'
                    
        # Extract data for the current benchmark
        BMA_data = BMA_df[BMA_column]
        if LSTM:
            LSTM_data = LSTM_df[LSTM_column]
        if data_type == 'BMA & RC & RCH & BC' or data_type == 'RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'BMA & RC & BC':
            RC_data = RC_df[RC_column]
            RCH_data = RCH_df[RCH_column]
            HYPER_data = HYPER_df[HYPER_column]

        # Create a DataFrame for the current benchmark
        if ce == 'cal':
            plot_df_cal = pd.DataFrame({})
            plot_df_cal['BMA'] = BMA_data
            if data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC'or data_type == 'BMA & RC & BC':
                if LSTM:
                    plot_df_cal['LSTM'] = LSTM_data
                plot_df_cal['RC'] = RC_data
                plot_df_cal['RCH'] = RCH_data
                plot_df_cal['BC'] = HYPER_data
            if data_type == 'RCH & BC':
                plot_df_cal['RCH'] = RCH_data
                plot_df_cal['BC'] = HYPER_data
                if LSTM:
                    plot_df_cal['LSTM'] = LSTM_data
            if data_type == 'RC & RCH & BC':
                plot_df_cal['RC'] = RC_data
                plot_df_cal['RCH'] = RCH_data
                plot_df_cal['BC'] = HYPER_data

            for model_name in model_list:
                MARRMoT_file_path = f'hyper/out/{loc}/MARRMoT_nocal/{model_name}_results_{ce}.csv'
                MARRMoT_df = pd.read_csv(MARRMoT_file_path)
                MARRMoT_column = f'{model_name}_{benchmark}_{ce}'
                MARRMoT_data = MARRMoT_df[MARRMoT_column]
                
                plot_df_cal[model_name] = MARRMoT_data
        else:
            plot_df_eva = pd.DataFrame({})
            plot_df_eva['BMA'] = BMA_data
            if data_type == 'BMA & RC & RCH & BC' or data_type == 'RC & RCH & BC' or data_type == 'BMA & RC & BC':
                plot_df_eva['RC'] = RC_data
                if LSTM:
                    plot_df_eva['LSTM'] = LSTM_data
                plot_df_eva['RCH'] = RCH_data
                plot_df_eva['BC'] = HYPER_data
            if data_type == 'RCH & BC':
                plot_df_eva['RCH'] = RCH_data
                plot_df_eva['BC'] = HYPER_data
                if LSTM:
                    plot_df_eva['LSTM'] = LSTM_data
            if data_type == 'RC & RCH & BC':
                plot_df_eva['RC'] = RC_data
                plot_df_eva['RCH'] = RCH_data
                plot_df_eva['BC'] = HYPER_data

            for model_name in model_list:
                MARRMoT_file_path = f'hyper/out/{loc}/MARRMoT_nocal/{model_name}_results_{ce}.csv'
                MARRMoT_df = pd.read_csv(MARRMoT_file_path)
                MARRMoT_column = f'{model_name}_{benchmark}_{ce}'
                MARRMoT_data = MARRMoT_df[MARRMoT_column]
                
                plot_df_eva[model_name] = MARRMoT_data

    create_box_plot(axes_box, plot_df_cal, plot_df_eva, benchmark, ce)
    #create_line_plot(axes_line, plot_df_cal, plot_df_eva, benchmark, ce)
    create_point_plot(axes_point, plot_df_cal, plot_df_eva, benchmark, ce)
    
    # Update global y-axis limits
    y_min, y_max = min(plot_df_cal.min().min(), plot_df_eva.min().min()), max(plot_df_cal.max().max(), plot_df_eva.max().max())
    global_y_min = min(global_y_min, y_min)
    global_y_max = max(global_y_max, y_max)
    
    if benchmark == 'NSE':
        global_y_min = max(global_y_min, -1)
        global_y_max = 1
    elif benchmark == 'KGE':
        global_y_min = max(global_y_min, -1)
        global_y_max = 1
    elif benchmark == 'E1':
        global_y_min = max(global_y_min, -1)
        global_y_max = 1
    elif benchmark == 'VE':
        global_y_max = 1
    elif benchmark == 'd':
        global_y_max = 1
    elif benchmark == 'RMSE' or benchmark == 'MAE':
        global_y_min = 0
        global_y_max = min(global_y_max, 6)

    
    # Set the same y-axis limits for both subplots
    all_axes = [axes_box[0], axes_point[0], axes_box[1], axes_point[1]]

    for ax in all_axes:
        ax.set_ylim(benchmark_limits[benchmark]['min'], benchmark_limits[benchmark]['max'])

    fig_box.tight_layout()
    #fig_line.tight_layout()
    fig_point.tight_layout()

    box_output_path = os.path.join(output_dir, f'box/{benchmark}_boxplot{buf}_{ce}.jpg')
    #line_output_path = os.path.join(output_dir, f'line/{benchmark}_lineplot{buf}_{ce}.jpg')
    point_output_path = os.path.join(output_dir, f'point/{benchmark}_pointplot{buf}_{ce}.jpg')

    fig_box.savefig(box_output_path)
    #fig_line.savefig(line_output_path)
    fig_point.savefig(point_output_path)

    plt.close(fig_box)
    #plt.close(fig_line)
    plt.close(fig_point)

    print(box_output_path)
print(f"saved to {output_dir}")
print("DONE!")
