# HYPER
Funato, M., & Sawada, Y. (2026). Multi-model ensemble and reservoir computing for river discharge prediction in ungauged basins. Journal of Geophysical Research: Machine Learning and Computation, 3, e2025JH001196. https://doi.org/10.1029/2025JH001196

## Prediction in Gauged Basins
### 00_HYPER: HYPER-BC model
HYPER model based on bias correction.  

Run the main_HYPER_BC.py

### 01_HYPER-RCH: HYPER-RCH model
Alternative HYPER model based on hybrid reservoir computing (Pathak et al., 2018).  

Run the main_RCH.py

## Prediction in Ungauged Basins
### 10_HYPER-BcReg 
HYPER-BC with regression based methods to estimate model weights for ungauged basins.  

main_BcReg_kfold.py: (data-rich scenario) for kfold analysis  
main_BcReg_random.py: (data-scarce scenario) for random selection of gauged basins including cases with limited number of gauged basins  
main_BcReg_region.py: (remote scenario) for prediction of ungauged basins using regional gauged basins  

### 11_HYPER-BcProx
HYPER-BC with spatial proximity based methods to estimate model weights for ungauged basins.  

main_BcProx_kfold.py: (data-rich scenario) for kfold analysis  
main_BcProx_random.py: (data-scarce scenario) for random selection of gauged basins including cases with limited number of gauged basins  
main_BcProx_region.py: (remote scenario) for prediction of ungauged basins using regional gauged basins  

## RC (benchmark)
### 20_RC
RC model without the use of BMA. 

Run the main_RC.py

## LSTM (benchmark)
### 90_LSTM
LSTM code for gauged basins, based on Kratzert et al., 2018  
    - Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and Herrnegger, M.: Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks, Hydrol. Earth Syst. Sci., 22, 6005-6022, https://doi.org/10.5194/hess-22-6005-2018, 2018. 

### 91_LSTM
LSTM code for ungauged basins, based on Kratzert et al., 2019  
    - Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning.Water Resources Research, 55. https://doi.org/10.1029/2019WR026065   
    
train_pub_kfold.sh: (data-rich scenario) for kfold analysis  
train_pub_random.sh: (data-scarce scenario) for random selection of gauged basins including cases with limited number of gauged basins  
train_pub_region.sh: (remote scenario) for prediction of ungauged basins using regional gauged basins  

## Analysis
### BMA & MARRMoT
main_BMA.py: for prediction using BMA only  
MARRMoT_model.py: for outputting the results of the uncalibrated MARRMoT models. 

### figures
Exp1_fig.py ~ Exp4_fig.py corresponds to the result analysis conducted within the paper.  

## data
### MERV-JP
The meteorological daily dataset and observed streamflow from Sawada & Okugawa, 2023 was used to run the uncalibrated version of the 43 MARRMoT models (Knoben, 2019) specified in the paper.   
    - Sawada, Y. and Okugawa, S.: Multi-model Ensemble for Robust Verification of hydrological modeling in Japan (MERV-Jp) (2.0), https://doi.org/10.5281/zenodo.8176305, 2023.  
    - Knoben, W. J. M.: wknoben/MARRMoT: MARRMoT_v1.3, , https://doi.org/10.5281/zenodo.3552961, 2019.  

### river_basin/dataset_JP
basin_data*.csv: the basin characteristics such as the topological, climatical, land form, land use, soil, and geological data, collected from MLIT data portal.  
    - MLIT: Geospatial Information, https://www.mlit.go.jp/tochi_fudousan_kensetsugyo/chirikukannjoho/tochi_fudousan_kensetsugyo_tk17_000001_00028.html, last access: 22 January 2025.  
pub_region_list_ver*.csv: The region column is used to classify the basins into regions.  

### river_basin
distance_matrix_v2*.csv: Created by mapping basins' location to QGIS and then using the "Distance Matrix" tool  
