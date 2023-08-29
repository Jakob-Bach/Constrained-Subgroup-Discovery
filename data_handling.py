"""Data handling

Functions for data I/O in the experimental pipeline (prediction datasets and experimental data).
"""


import pathlib
from typing import Generator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.model_selection


# Feature-part and target-part of a dataset are saved separately.
def load_dataset(dataset_name: str, directory: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(directory / (dataset_name + '_X.csv'))
    y = pd.read_csv(directory / (dataset_name + '_y.csv')).squeeze(axis='columns')
    assert isinstance(y, pd.Series)  # a DataFrame might cause errors somewhere in the pipeline
    return X, y


def save_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str, directory: pathlib.Path) -> None:
    X.to_csv(directory / (dataset_name + '_X.csv'), index=False)
    y.to_csv(directory / (dataset_name + '_y.csv'), index=False)


# List dataset names based on target-values_files.
def list_datasets(directory: pathlib.Path) -> Sequence[str]:
    return [file.name.split('_y.')[0] for file in list(directory.glob('*_y.*'))]


def load_dataset_overview(directory: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(directory / '_dataset_overview.csv')


def save_dataset_overview(dataset_overview: pd.DataFrame, directory: pathlib.Path) -> None:
    dataset_overview.to_csv(directory / '_dataset_overview.csv', index=False)


# Split a dataset for the experimental pipeline. Return the split indices.
def split_for_pipeline(X: pd.DataFrame, y: pd.Series, n_splits: int = 5)\
        -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True,
                                                       random_state=25)
    return splitter.split(X=X, y=y)


# Return the path of the file containing either complete experimental results or only a particular
# combination of dataset, fold, and subgroup-discovery method.
def get_results_file_path(directory: pathlib.Path, dataset_name: Optional[str] = None,
                          split_idx: Optional[int] = None, sd_name: Optional[str] = None) -> pathlib.Path:
    if (dataset_name is not None) and (split_idx is not None) and (sd_name is not None):
        return directory / (f'{dataset_name}_{split_idx}_{sd_name}_results.csv')
    return directory / '_results.csv'


# Load either complete results or only a particular combi of dataset, fold, and subgroup discovery.
def load_results(directory: pathlib.Path, dataset_name: Optional[str] = None,
                 split_idx: Optional[int] = None, sd_name: Optional[str] = None) -> pd.DataFrame:
    results_file = get_results_file_path(directory=directory, dataset_name=dataset_name,
                                         split_idx=split_idx, sd_name=sd_name)
    if results_file.exists():
        return pd.read_csv(results_file)
    # If particular results file does not exist, just grab and merge all results in the directory:
    return pd.concat([pd.read_csv(x) for x in directory.glob('*_results.*')], ignore_index=True)


def save_results(results: pd.DataFrame, directory: pathlib.Path, dataset_name: Optional[str] = None,
                 split_idx: Optional[int] = None,  sd_name: Optional[str] = None) -> None:
    results_file = get_results_file_path(directory=directory, dataset_name=dataset_name,
                                         split_idx=split_idx, sd_name=sd_name)
    results.to_csv(results_file, index=False)
