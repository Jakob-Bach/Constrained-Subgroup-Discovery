"""Prepare demo datasets

Script to minimally preprocess and save a few datasets from scikit-learn (toy and synthetically
generated), so they can be used in the experimental pipeline.
The actual experiments for the paper use different datasets.

Usage: python -m prepare_demo_datasets --help
"""


import argparse
import itertools
import pathlib

import pandas as pd
import sklearn.datasets
import tqdm

import data_handling


SYN_DATASETS_N_INSTANCES = [100, 200, 400]
SYN_DATASETS_N_FEATURES = [10, 20, 40]
TOY_DATASETS_NAMES = ['breast_cancer', 'digits', 'iris', 'wine']


# Main routine: Pre-process and save (to "data_dir") datasets from scikit-learn.
def prepare_datasets(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Dataset directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if any(data_dir.iterdir()):
        print('Dataset directory is not empty. Files might be overwritten, but not deleted.')
    dataset_overview = []  # gets a subset of the columns used in PMLB's integrated dataset summary

    print('Saving toy datasets ...')
    for dataset_name in tqdm.tqdm(TOY_DATASETS_NAMES):
        X, y = getattr(sklearn.datasets, 'load_' + dataset_name)(as_frame=True, return_X_y=True)
        y = (y == 0).astype(int)  # binarize multi-class targets by predicting Class 0
        data_handling.save_dataset(X=X, y=y, dataset_name=dataset_name, directory=data_dir)
        dataset_overview.append({'dataset': dataset_name, 'n_instances': X.shape[0],
                                 'n_features': X.shape[1]})

    print('Saving synthetically generated datasets ...')
    for n_instances, n_features in tqdm.tqdm(list(itertools.product(SYN_DATASETS_N_INSTANCES,
                                                                    SYN_DATASETS_N_FEATURES))):
        X, y = sklearn.datasets.make_classification(n_samples=n_instances, n_features=n_features,
                                                    n_classes=2, random_state=25)
        X = pd.DataFrame(X, columns=[f'Feature_{j + 1}' for j in range(n_features)])
        y = pd.Series(y, name='target')
        dataset_name = f'syn_{n_instances}_{n_features}'
        data_handling.save_dataset(X=X, y=y, dataset_name=dataset_name, directory=data_dir)
        dataset_overview.append({'dataset': dataset_name, 'n_instances': X.shape[0],
                                 'n_features': X.shape[1]})

    data_handling.save_dataset_overview(pd.DataFrame(dataset_overview), directory=data_dir)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves small demo datasets from scikit-learn, prepares them for the ' +
        'experimental pipeline, and stores them in the specified directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/datasets/',
                        dest='data_dir', help='Directory to store prediction datasets.')
    print('Dataset preparation started.')
    prepare_datasets(**vars(parser.parse_args()))
    print('Datasets prepared and saved.')
