"""Prepare competitor-runtime datasets

Script to save a few small datasets from PMLB (subset of datasets from main experiments) or
scikit-learn, so they can be used in the pipeline for competitor-runtime experiments.

Usage: python -m prepare_competitor_runtime_datasets --help
"""


import argparse
import pathlib

import pandas as pd
import pmlb
import sklearn.datasets
import tqdm

import data_handling


# Datasets used here are a subset from main experiments, with m <= 500 and n <= 50
PMLB_DATASETS = ['backache', 'horse_colic', 'ionosphere', 'spect', 'spectf']


# Main routine: Download, pre-process, and save (to "data_dir") datasets.
def prepare_datasets(data_dir: pathlib.Path, source: str) -> None:
    if not data_dir.is_dir():
        print('Dataset directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Dataset directory is not empty. Files might be overwritten, but not deleted.')

    if source == 'pmlb':  # multiple small datasets
        dataset_overview = pmlb.dataset_lists.df_summary
        dataset_overview = dataset_overview[dataset_overview['dataset'].isin(PMLB_DATASETS)]
        assert len(dataset_overview) == len(PMLB_DATASETS)  # check for typos
        print('Downloading and saving PMLB datasets ...')
        for dataset_name in tqdm.tqdm(dataset_overview['dataset']):
            dataset = pmlb.fetch_data(dataset_name=dataset_name, dropna=False)
            assert dataset.notna().all().all()  # datasets we chose don't contain missing values
            X = dataset.drop(columns='target')
            minority_class = dataset['target'].value_counts().idxmin()
            y = (dataset['target'] == minority_class).astype(int)  # not all targets are (0, 1)
            data_handling.save_dataset(X=X, y=y, dataset_name=dataset_name, directory=data_dir)
    elif source == 'sklearn':  # just one (very small) dataset
        dataset_name = 'iris'
        X, y = getattr(sklearn.datasets, 'load_' + dataset_name)(as_frame=True, return_X_y=True)
        y = (y == 0).astype(int)  # binarize multi-class targets by predicting Class 0
        data_handling.save_dataset(X=X, y=y, dataset_name=dataset_name, directory=data_dir)
        dataset_overview = pd.DataFrame({'dataset': [dataset_name], 'n_instances': [X.shape[0]],
                                         'n_features': [X.shape[1]]})
    else:
        raise ValueError(f'Illegal value for dataset source: "{source}"')
    data_handling.save_dataset_overview(dataset_overview=dataset_overview, directory=data_dir)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves datasets from PMLB or scikit-learn, prepares them for the' +
        'competitor-runtime experiments, and stores them in the specified directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path,
                        default='data/competitor-runtime-datasets/',
                        dest='data_dir', help='Directory to store prediction datasets.')
    parser.add_argument('-s', '--source', type=str, default='pmlb', choices=['sklearn', 'pmlb'],
                        dest='source', help='Source to fetch datasets from.')
    print('Competitor-runtime dataset preparation started.')
    prepare_datasets(**vars(parser.parse_args()))
    print('Competitor-runtime datasets prepared and saved.')
