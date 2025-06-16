import os
import numpy as np
import pandas as pd

loc, ver = "JP", 2

start_date_cal = '1993-01-01'
end_date_cal = '2000-12-31'
start_date_eva = '2001-01-01'
end_date_eva = '2006-12-31'

attribute_dir = f'hyper/data/river_basin/dataset_{loc}'

ver_name = "ver1_1" if ver == 1 else "ver2_0"
if loc == "JP" and ver == 2:
    file_tot_num = 87

varssim_dir = f"hyper/data/MERVJP/varssim_nocal/{ver_name}"

output_dir = f"hyper/data/MERVJP/stats"
os.makedirs(output_dir, exist_ok=True)

def file_name(input_num, total_len):
    car_len = len(str(input_num))
    zero_len = total_len - car_len
    return '0' * zero_len + str(input_num)

def load_data(file_num):
    df = pd.read_csv(f"{varssim_dir}/varssim{file_name(file_num, 3)}.csv")
    
    if loc == "JP":
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a date range that covers the entire period
    full_date_range = pd.date_range(start=start_date_cal, end=end_date_eva)
    
    # Reindex the DataFrame to this date range, filling missing values with NaNs
    df = df.reindex(full_date_range)
    
    return df

# Initialize an empty DataFrame to store combined data
combined_df = pd.DataFrame()

for file_num in range(1, file_tot_num + 1):
    df = load_data(file_num)
    combined_df = pd.concat([combined_df, df])

print(combined_df.head())
print(combined_df.shape)

# Calculate mean and std for each column
mean_std_df = combined_df.aggregate(['mean', 'std']).transpose()
mean_std_df.columns = ['mean', 'std']

# Save the mean and std values to a single CSV file
mean_std_df.to_csv(f"{output_dir}/flow_combined_mean_std.csv")
