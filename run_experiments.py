"""Run experiments

Script to run the complete experimental pipeline. Should be run after dataset preparation, as the
experiments require prediction datasets as inputs. Saves its results for evaluation. If some
results already exist, only runs the missing experimental tasks.

Usage: python -m run_experiments --help
"""


import argparse
import itertools
import multiprocessing
import pathlib
from typing import Any, Dict, Optional, Sequence, Type

import pandas as pd
import tqdm

import data_handling
import sd


# Different components of the experimental design.
N_FOLDS = 5  # cross-validation
SUBGROUP_DISCOVERY_TYPES = [sd.MIPSubgroupDiscoverer, sd.SMTSubgroupDiscoverer]


# Define experimental tasks (for parallelization) as cross-product of datasets (from "data_dir"),
# cross-validation folds, and subgroup-discovery methods. Only return tasks for which there is no
# results file in "results_dir". Provide a dictionary for calling "evaluate_experimental_task()".
def define_experimental_tasks(data_dir: pathlib.Path,
                              results_dir: pathlib.Path) -> Sequence[Dict[str, Any]]:
    experimental_tasks = []
    dataset_names = data_handling.list_datasets(directory=data_dir)
    for dataset_name, split_idx, subgroup_discovery_type in itertools.product(
            dataset_names, range(N_FOLDS), SUBGROUP_DISCOVERY_TYPES):
        results_file = data_handling.get_results_file_path(
            directory=results_dir, dataset_name=dataset_name, split_idx=split_idx,
            sd_name=subgroup_discovery_type.__name__)
        if not results_file.exists():
            experimental_tasks.append(
                {'dataset_name': dataset_name, 'data_dir': data_dir, 'results_dir': results_dir,
                 'split_idx': split_idx, 'subgroup_discovery_type': subgroup_discovery_type})
    return experimental_tasks


# Evaluate one subgroup-discovery method on one split of one dataset. To this end, read in the
# dataset with the "dataset_name" from the "data_dir" and extract the "split_idx"-th split.
# "subgroup_discovery_type" is a class representing the subgroup-discovery method.
# Return a DataFrame with various evaluation metrics, including parametrization of the search for
# subgroups, runtime, and prediction performance. Additionally, save this data to "results_dir".
def evaluate_experimental_task(
        dataset_name: str, data_dir: pathlib.Path, results_dir: pathlib.Path, split_idx: int,
        subgroup_discovery_type: Type[sd.SubgroupDiscoverer]) -> pd.DataFrame:
    X, y = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    subgroup_discoverer = subgroup_discovery_type()
    train_idx, test_idx = list(data_handling.split_for_pipeline(X=X, y=y, n_splits=N_FOLDS))[split_idx]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    results = [subgroup_discoverer.evaluate(X_train=X_train, y_train=y_train,
                                            X_test=X_test, y_test=y_test)]
    results = pd.DataFrame(results)
    results['sd_name'] = subgroup_discovery_type.__name__
    results['dataset_name'] = dataset_name
    results['split_idx'] = split_idx
    data_handling.save_results(results=results, directory=results_dir, dataset_name=dataset_name,
                               split_idx=split_idx, sd_name=subgroup_discovery_type.__name__)
    return results


# Main-routine: Run complete experimental pipeline. To this end, read datasets from "data_dir",
# save results to "results_dir". "n_processes" controls parallelization (over datasets,
# cross-validation folds, and subgroup-discovery methods).
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
        description='Runs complete experimental pipeline except tasks that already have results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/datasets/', dest='data_dir',
                        help='Directory with input data, i.e., prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/', dest='results_dir',
                        help='Directory for output data, i.e., experimental results.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for parallelization (default: all cores).')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
