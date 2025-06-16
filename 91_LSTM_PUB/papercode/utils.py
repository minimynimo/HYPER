"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import sys
from pathlib import Path, PosixPath
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .datasets import CamelsTXT


def create_h5_files(out_file: PosixPath,
                    basins: List,
                    dates: List,
                    with_basin_str: bool = True,
                    seq_length: int = 270,
                    varssim_dir: str = None,
                    loc: str = None):
    """[summary]
    
    Parameters
    ----------
    out_file : PosixPath
        Path of the location, where the hdf5 file should be stored
    basins : List
        List containing the 8-digit USGS gauge id
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    with_basin_str : bool, optional
        If True, stores for each sample the corresponding USGS gauged id, by default True
    seq_length : int, optional
        Length of the requested input sequences., by default 270
    varssim_dir : str, optional
        Directory containing the varssim files, by default None
    loc : str, optional
        Location identifier, by default None
    
    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset(
            'input_data',
            shape=(0, seq_length, 3),
            maxshape=(None, seq_length, 3),
            chunks=True,
            dtype=np.float32,
            compression='gzip')
        target_data = out_f.create_dataset(
            'target_data',
            shape=(0, 1),
            maxshape=(None, 1),
            chunks=True,
            dtype=np.float32,
            compression='gzip')

        q_stds = out_f.create_dataset(
            'q_stds',
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.float32,
            compression='gzip',
            chunks=True)

        if with_basin_str:
            sample_2_basin = out_f.create_dataset(
                'sample_2_basin',
                shape=(0,),
                maxshape=(None,),
                dtype="S10",
                compression='gzip',
                chunks=True)

        for file_num, basin in enumerate(basins, start=1):  # Use file_num for varssim files
            try:
                dataset = CamelsTXT(
                    basin=basin,
                    is_train=True,
                    seq_length=seq_length,
                    dates=dates,
                    file_num=file_num,
                    varssim_dir=varssim_dir,
                    loc=loc)
            except Exception as e:
                print(f"Warning: Could not create data for basin {basin}. Error: {e}")
                continue

            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples

            # store input and output samples
            input_data.resize((total_samples, seq_length, 3))
            target_data.resize((total_samples, 1))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :] = dataset.y

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            if with_basin_str:
                sample_2_basin.resize((total_samples,))
                str_arr = np.array([str(basin).encode("ascii", "ignore")] * num_samples)  # Convert basin to string
                sample_2_basin[-num_samples:] = str_arr

            out_f.flush()


def get_basin_list() -> List:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path(__file__).absolute().parent.parent / "data/basin_list.txt"
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins
