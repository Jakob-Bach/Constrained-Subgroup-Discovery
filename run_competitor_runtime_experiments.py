"""Run competitor-runtime experiments

Script to run the competitor-runtime experiments. Should be run after dataset preparation, as the
experiments require prediction datasets as inputs. Saves its results for evaluation. If some
results already exist, only runs the missing experimental tasks (combinations of dataset,
cross-validation fold, and subgroup-discovery method).

Usage: python -m run_competitor_runtime_experiments --help
"""


import argparse
import itertools
import multiprocessing
import pathlib
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import tqdm

import runtime_competitors
import data_handling


# Different components of the experimental design.
N_FOLDS = 5  # number of cross-validation folds
CARDINALITIES = [1, 2, 3, 4, 5, None]  # maximum number of features selected in subgroup
SD_METHODS = {
    'csd.Beam': runtime_competitors.CSD_BeamSearch,
    'csd.SMT': runtime_competitors.CSD_SMT,
}


# Define experimental tasks (for parallelization) as cross-product of datasets (from "data_dir"),
# cross-validation folds, and subgroup-discovery methods. Provide a dictionary for calling
# "evaluate_experimental_task()", only including tasks for which there is no results file in
# "results_dir".
def define_experimental_tasks(data_dir: pathlib.Path,
                              results_dir: pathlib.Path) -> Sequence[Dict[str, Any]]:
    experimental_tasks = []
    dataset_names = data_handling.list_datasets(directory=data_dir)
    for dataset_name, split_idx, sd_name in itertools.product(
            dataset_names, range(N_FOLDS), SD_METHODS):
        results_file = data_handling.get_results_file_path(
            directory=results_dir, dataset_name=dataset_name, split_idx=split_idx, sd_name=sd_name)
        if not results_file.exists():
            experimental_tasks.append(
                {'dataset_name': dataset_name, 'data_dir': data_dir, 'results_dir': results_dir,
                 'split_idx': split_idx, 'sd_name': sd_name})
    return experimental_tasks


# Evaluate one subgroup-discovery method on one split of one dataset. To this end, read in the
# dataset with the "dataset_name" from the "data_dir" and extract the "split_idx"-th split.
# Evaluate different feature-cardinality thresholds for the method identified by "sd_name".
# Return a DataFrame with experimental factors and runtime of subgroup discovery.
# Additionally, save this data to "results_dir".
def evaluate_experimental_task(
        dataset_name: str, data_dir: pathlib.Path, results_dir: pathlib.Path, split_idx: int,
        sd_name: str) -> pd.DataFrame:
    X, y = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    subgroup_discoverer = SD_METHODS[sd_name]()
    train_idx, _ = list(data_handling.split_for_pipeline(X=X, y=y, n_splits=N_FOLDS))[split_idx]
    results = []
    for k in CARDINALITIES:
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        runtime = subgroup_discoverer.evaluate_runtime(X=X_train, y=y_train, k=k)
        results.append({
            'dataset_name': dataset_name,
            'split_idx': split_idx,
            'sd_name': sd_name,
            'param.k': k,
            'fitting_time': runtime
        })
    results = pd.DataFrame(results)
    data_handling.save_results(results=results, directory=results_dir, dataset_name=dataset_name,
                               split_idx=split_idx, sd_name=sd_name)
    return results


# Main routine: Run competitor-runtime experiments. To this end, read datasets from "data_dir",
# save results to "results_dir". "n_processes" controls parallelization (over datasets,
# cross-validation folds, and subgroup-discovery methods); by default, all cores used.
def run_experiments(data_dir: pathlib.Path, results_dir: pathlib.Path,
                    n_processes: Optional[int] = None) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Dataset directory does not exist.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if any(results_dir.iterdir()):
        print('Results directory is not empty. Only missing experiments will be run.')
    experimental_tasks = define_experimental_tasks(data_dir=data_dir, results_dir=results_dir)
    progress_bar = tqdm.tqdm(total=len(experimental_tasks))
    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_experimental_task, kwds=task,
                                        callback=lambda x: progress_bar.update())
               for task in experimental_tasks]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    results = data_handling.load_results(directory=results_dir)  # merge individual results files
    data_handling.save_results(results, directory=results_dir)


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs competitor-runtime experiments except tasks that already have results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, dest='data_dir',
                        default='data/competitor-runtime-datasets/',
                        help='Directory with input data, i.e., prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, dest='results_dir',
                        default='data/competitor-runtime-results/',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for parallelization (default: all cores).')
    print('Competitor-runtime experiments started.')
    run_experiments(**vars(parser.parse_args()))
    print('Competitor-runtime experiments executed successfully.')
