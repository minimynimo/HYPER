"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import add_basin_attributes, rescale_features
from papercode.lstm import LSTM
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

import os
import time

loc, ver = "JP", 2

ver_name = "ver1_1" if ver == 1 else "ver2_0"
##Change datautils.py loc as well


output_dir = f'out/{loc}/LSTM_PUB/random/data'
varssim_dir = f"data/MERVJP/varssim_nocal/{ver_name}"

rundir = f'out/{loc}/LSTM_PUB/random'
os.makedirs(rundir, exist_ok=True)

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 2000,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('01011993', format='%d%m%Y'),
    'train_end': pd.to_datetime('31122000', format='%d%m%Y'),
    'val_start': pd.to_datetime('01012001', format='%d%m%Y'),
    'val_end': pd.to_datetime('31122006', format='%d%m%Y')
}



if loc == "JP":
    file_tot_num = 87
    attribute_size = 196
    output_mean = np.array([3.8526752089320553])
    test_basins_list = [4,8,11,18,24,28,32,40,45,50,54,59,65,70,77,82,84]
file_num_list = list(map(str, range(1, file_tot_num + 1)))
###############
# Prepare run #
###############


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', 
        choices=["train", "evaluate", "eval_robustness", "create_splits", "create_random_train_test"],  # Added create_random_train_test
    )
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument(
        '--run_dir', 
        type=str, 
        default=rundir,
        help="For evaluation mode. Path to run directory.")
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help="User-selected GPU ID - if none chosen, will default to cpu")
    parser.add_argument(
        '--cache_data', type=str2bool, default=True, help="If True, loads all data into memory")
    parser.add_argument(
        '--num_workers', type=int, default=12, help="Number of parallel threads for data loading")
    parser.add_argument(
        '--no_static',
        type=str2bool,
        default=False,
        help="If True, trains LSTM without static features")
    parser.add_argument(
        '--concat_static',
        type=str2bool,
        default=True,
        help="If True, train LSTM with static feats concatenated at each time step")
    parser.add_argument(
        '--use_mse',
        type=str2bool,
        default=True,  # Changed from false 
        help="If True, uses mean squared error as loss function.")
    parser.add_argument(
        '--split_file',
        type=str,
        default=None,
        help="Path to file created from the `create_splits` function.")
    parser.add_argument(
        '--train_basin_int', 
        type=int, 
        required=True,  # Make this argument required
        help="Number of training basins to select randomly."
    )
    cfg = vars(parser.parse_args())

    os.makedirs(rundir, exist_ok=True)
    #create all subdirectories that are under rundir
    os.makedirs(rundir + '/reports', exist_ok=True)
    os.makedirs(rundir + '/data', exist_ok=True)

    # Validation checks
    if (cfg["mode"] in ["train", "create_splits"]) and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # GPU selection
    if cfg["gpu"] >= 0:
        device = f"cuda:{cfg['gpu']}"
    else:
        device = 'cpu'

    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_train{cfg["train_basin_int"]}_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(rundir) / "runs" / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the basin IDs or file numbers

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_basin_attributes(db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(
        out_file=cfg["train_file"],
        basins=basins,  # Use the provided basins list
        dates=[cfg["train_start"], cfg["train_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"],
        varssim_dir=varssim_dir,
        loc=loc)  # Pass the location identifier

    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        self.lstm = LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # Log the start of the training process
    print(f"Starting training with seed={cfg['seed']} and train_basin_int={cfg['train_basin_int']}")

    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["split_file"] is not None:
        split_file_path = Path(cfg["split_file"])
        if not split_file_path.is_absolute():
            split_file_path = Path(__file__).absolute().parent / cfg["split_file"]

        # Wait for the split file to be created
        wait_time = 0
        while not split_file_path.exists():
            if wait_time > 300:  # Timeout after 5 minutes
                raise FileNotFoundError(f"DEBUG: Split file not found at {split_file_path} after waiting.")
            time.sleep(5)
            wait_time += 5

        print(f"DEBUG: Loading split file from {split_file_path}")
        with split_file_path.open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits["train"]  # Use "train" key directly
    else:
        basins = file_num_list

    # Convert basins (file_num) to strings if necessary
    basins = [str(basin) for basin in basins]
    print(f"Training basins: {basins}")

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare PyTorch DataLoader
    try:
        ds = CamelsH5(
            h5_file=cfg["train_file"],
            basins=basins,
            db_path=cfg["db_path"],
            concat_static=cfg["concat_static"],
            cache=cfg["cache_data"],
            no_static=cfg["no_static"])
        loader = DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    except Exception as e:
        raise RuntimeError(f"Error while preparing DataLoader: {e}")

    # create model and optimizer
    input_size_dyn = 3 if (cfg["no_static"] or not cfg["concat_static"]) else attribute_size
    model = Model(
        input_size_dyn=input_size_dyn,
        hidden_size=cfg["hidden_size"],
        initial_forget_bias=cfg["initial_forget_gate_bias"],
        dropout=cfg["dropout"],
        concat_static=cfg["concat_static"],
        no_static=cfg["no_static"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        try:
            train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])
        except Exception as e:
            raise RuntimeError(f"DEBUG: Error during training epoch {epoch}: {e}")

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))



def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        x, y, q_stds = data
        # Check if x includes nan values

        x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
        predictions = model(x)[0]
        
        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation con
    cal_eva : str
        If True, evaluates the model on the training data, else on the validation data
        
fig

    cal_eva : str
        If True, evaluates the model on the training data, else on the validation data
        
    """
    cal_str_list = ["cal", "val"]
    #cal_str_list = ["val", "cal"]

    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    """if user_cfg["split_file"] is not None:
        with Path(user_cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits["test"]  # Use "test" key directly
    else:
        basins = get_basin_list()"""
    
    basins = [str(basin) for basin in test_basins_list]

    print("in evaluate")
    print(basins)

    # get attribute means/stds from trainings dataset
    train_file = user_cfg["run_dir"] / "data/train/train_data.h5"
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    ds_train = CamelsH5(
        h5_file=train_file, db_path=db_path, basins=basins, concat_static=run_cfg["concat_static"])
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()

    for cal_eva in cal_str_list:

        # create model
        input_size_dyn = 3 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else attribute_size
        model = Model(
            input_size_dyn=input_size_dyn,
            hidden_size=run_cfg["hidden_size"],
            dropout=run_cfg["dropout"],
            concat_static=run_cfg["concat_static"],
            no_static=run_cfg["no_static"]).to(DEVICE)

        # load trained model
        weight_file = user_cfg["run_dir"] / f'model_epoch30.pt'
        model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

        if cal_eva == "cal":
            #because the training data only exists from train_start, need to offset the sequence length to predict properly
            train_start_date = GLOBAL_SETTINGS["train_start"] + pd.DateOffset(days=run_cfg["seq_length"] - 1)
            date_range = pd.date_range(start=train_start_date, end=GLOBAL_SETTINGS["train_end"])
        elif cal_eva == "val":
            date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])

        results = {}
        for basin in tqdm(basins):
            if cal_eva == "cal":
                ds_test = CamelsTXT(
                    basin=basin,
                    dates=[train_start_date, GLOBAL_SETTINGS["train_end"]],
                    is_train=False,
                    seq_length=run_cfg["seq_length"],
                    with_attributes=True,
                    attribute_means=means,
                    attribute_stds=stds,
                    file_num=basin,
                    varssim_dir=varssim_dir,
                    loc=loc,
                    concat_static=run_cfg["concat_static"],
                    db_path=db_path)
            elif cal_eva == "val":
                ds_test = CamelsTXT(
                    basin=basin,
                    dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                    is_train=False,
                    seq_length=run_cfg["seq_length"],
                    with_attributes=True,
                    attribute_means=means,
                    attribute_stds=stds,
                    file_num=basin,
                    varssim_dir=varssim_dir,
                    loc=loc,
                    concat_static=run_cfg["concat_static"],
                    db_path=db_path)
                
            loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)

            preds, obs = evaluate_basin(model, loader)

            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)

            results[basin] = df

            _store_results(user_cfg, run_cfg, results, cal_eva=cal_eva)


def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(preds.numpy(), variable='output')
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs


def _store_results(user_cfg: Dict, run_cfg: Dict, results, cal_eva: str):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.
    cal_eva : str
        Indicates whether the results are for calibration ("cal") or validation ("val").
    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"lstm_no_static_seed{run_cfg['seed']}_{'cal' if cal_eva == 'cal' else 'val'}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}_{'cal' if cal_eva == 'cal' else 'val'}.p"
        else:
            file_name = user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}_{'cal' if cal_eva == 'cal' else 'val'}.p"

    with file_name.open('wb') as fp:
        pickle.dump(results, fp)

    #print(f"Successfully stored results at {file_name}")


######################################
# Select random train and test basin #
######################################

def create_random_train_test(cfg: dict):
    """Create random train-test selection for evaluation.

    Takes basins test_basins_list and sets them as fixed test basins. The rest of the basins are used for random selection of training basins.
    A single random selection of training basins is created for each seed. The remaining basins are used as test basins.
    The result is stored into a dictionary, that contains a `train` and a `test` key, which contain the list of train and test basins.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the user entered evaluation config.
    """
    basins = file_num_list

    # Select the basins that are used for random selection
    non_fixed_basins = [b for b in basins if b not in test_basins_list]

    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(f'{output_dir}/random_splits_train{cfg["train_basin_int"]}_seed{cfg["seed"]}.p') 

    # Overwrite the output file if it already exists
    if output_file.is_file():
        output_file.unlink()

    # Set random seed for reproducibility
    np.random.seed(cfg["seed"])

    # Create a dictionary to store the selection
    splits = {}

    # Generate a single random selection for the given seed
    train_basins = np.random.choice(non_fixed_basins, cfg["train_basin_int"], replace=False).tolist()
    test_basins = [b for b in basins if b not in train_basins]

    # Store the training and testing basins
    splits['train'] = train_basins
    splits['test'] = test_basins

    # Save the split to a file
    with output_file.open('wb') as fp:
        pickle.dump(splits, fp)

    print(f"Stored dictionary with train-test selection at {output_file}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
