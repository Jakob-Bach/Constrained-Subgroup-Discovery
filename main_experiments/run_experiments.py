"""Run experiments

Script to run the complete experimental pipeline. Should be run after dataset preparation, as the
experiments require prediction datasets as inputs. Saves its results for evaluation. If some
results already exist, only runs the missing experimental tasks (combinations of dataset,
cross-validation fold, and subgroup-discovery method).

Usage: python -m run_experiments --help
"""


import argparse
import itertools
import multiprocessing
import pathlib
import time
from typing import Any, Dict, Optional, Sequence, Type, Union

import pandas as pd
import tqdm

import data_handling
import csd
import sd4py_methods


# Different components of the experimental design.
N_FOLDS = 5  # number of cross-validation folds
SOLVER_TIMEOUTS = [2 ** x for x in range(12)]  # in seconds
CARDINALITIES = [1, 2, 3, 4, 5, None]  # maximum number of features selected in subgroup
ALT_CARDINALITIES = [3]  # cardinalities for which alternatives should be searched
ALT_NUMBER = 5  # number of alternatives if alteratives should be searched


# Define a list of subgroup-discovery methods for "define_experimental_tasks()", each element
# comprising the method itself and a list of (dictionaries containing) hyperparameter combinations
# used to initialize the method.
def define_sd_methods() -> Sequence[Dict[str, Union[csd.SubgroupDiscoverer, Dict[str, Any]]]]:
    card_args = [{'k': k} for k in CARDINALITIES]
    smt_args = []
    for timeout, k in itertools.product(SOLVER_TIMEOUTS, CARDINALITIES):
        if (k in ALT_CARDINALITIES) and timeout == max(SOLVER_TIMEOUTS):
            for tau_abs in range(1, k + 1):
                smt_args.append({'timeout': timeout, 'k': k, 'a': ALT_NUMBER, 'tau_abs': tau_abs})
        else:
            smt_args.append({'timeout': timeout, 'k': k})
    beam_args = []
    for k in CARDINALITIES:
        if k in ALT_CARDINALITIES:
            for tau_abs in range(1, k + 1):
                beam_args.append({'k': k, 'a': ALT_NUMBER, 'tau_abs': tau_abs})
        else:
            beam_args.append({'k': k})
    return [
        {'sd_name': 'SMT', 'sd_type': csd.SMTSubgroupDiscoverer, 'sd_args_list': smt_args},
        {'sd_name': 'MORS', 'sd_type': csd.MORSSubgroupDiscoverer, 'sd_args_list': card_args},
        {'sd_name': 'Random', 'sd_type': csd.RandomSubgroupDiscoverer, 'sd_args_list': card_args},
        {'sd_name': 'PRIM', 'sd_type': csd.PRIMSubgroupDiscoverer, 'sd_args_list': card_args},
        {'sd_name': 'BI', 'sd_type': csd.BestIntervalSubgroupDiscoverer, 'sd_args_list': card_args},
        {'sd_name': 'Beam', 'sd_type': csd.BeamSearchSubgroupDiscoverer, 'sd_args_list': beam_args},
        {'sd_name': 'BSD', 'sd_type': sd4py_methods.BSDSubgroupDiscoverer, 'sd_args_list': card_args},
        {'sd_name': 'SD-Map', 'sd_type': sd4py_methods.SDMapSubgroupDiscoverer, 'sd_args_list': card_args}
    ]


SD4PY_METHODS = ['BSD', 'SD-Map']
SD4PY_TIMEOUT_DATASETS = ['coil2000', 'Hill_Valley_with_noise', 'sonar', 'spambase', 'spectf']


# While the exhaustive search methods from the package "SD4Py" are generally fast, they did not
# finish the unconstrained setting within two days on some datasets. For these, we manually exclude
# the unconstrained setting.
def filter_infeasible_args(dataset_name: str, sd_name: str,
                           sd_args_list: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    if (dataset_name in SD4PY_TIMEOUT_DATASETS) and (sd_name in SD4PY_METHODS):
        sd_args_list = [args for args in sd_args_list if args['k'] is not None]
    return sd_args_list


# Define experimental tasks (for parallelization) as cross-product of datasets (from "data_dir"),
# cross-validation folds, and subgroup-discovery methods (each including several hyperparameter
# settings). Provide a dictionary for calling "evaluate_experimental_task()", only including tasks
# for which there is no results file in "results_dir".
def define_experimental_tasks(data_dir: pathlib.Path,
                              results_dir: pathlib.Path) -> Sequence[Dict[str, Any]]:
    experimental_tasks = []
    sd_method_settings = define_sd_methods()
    dataset_names = data_handling.list_datasets(directory=data_dir)
    for dataset_name, split_idx, sd_method_setting in itertools.product(
            dataset_names, range(N_FOLDS), sd_method_settings):
        results_file = data_handling.get_results_file_path(
            directory=results_dir, dataset_name=dataset_name, split_idx=split_idx,
            sd_name=sd_method_setting['sd_name'])
        if not results_file.exists():
            experimental_tasks.append(
                {'dataset_name': dataset_name, 'data_dir': data_dir, 'results_dir': results_dir,
                 'split_idx': split_idx, **sd_method_setting})
    return experimental_tasks


# Evaluate one subgroup-discovery method on one split of one dataset. To this end, read in the
# dataset with the "dataset_name" from the "data_dir" and extract the "split_idx"-th split.
# "sd_type" is a class representing the subgroup-discovery method, while "sd_name" is an arbitrary
# (user-defined) name for the method and "sd_args_list" is a list of hyperparameter combinations
# used to initialize the method, which will be evaluated sequentially.
# Return a DataFrame with various evaluation metrics, including parametrization of the search for
# subgroups, runtime, and subgroup quality. Additionally, save this data to "results_dir".
def evaluate_experimental_task(
        dataset_name: str, data_dir: pathlib.Path, results_dir: pathlib.Path, split_idx: int,
        sd_name: str, sd_type: Type[csd.SubgroupDiscoverer], sd_args_list: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    X, y = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    train_idx, test_idx = list(data_handling.split_for_pipeline(X=X, y=y, n_splits=N_FOLDS))[split_idx]
    results = []
    sd_args_list = filter_infeasible_args(dataset_name=dataset_name, sd_name=sd_name,
                                          sd_args_list=sd_args_list)  # some settings take too long
    for sd_args in sd_args_list:
        subgroup_discoverer = sd_type(**sd_args)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        result = subgroup_discoverer.evaluate(X_train=X_train, y_train=y_train,
                                              X_test=X_test, y_test=y_test)  # returns DataFrame
        result['dataset_name'] = dataset_name
        result['split_idx'] = split_idx
        result['sd_name'] = sd_name
        for key, value in sd_args.items():  # save all hyperparameter values
            result[f'param.{key}'] = value
        results.append(result)
    results = pd.concat(results)
    data_handling.save_results(results=results, directory=results_dir, dataset_name=dataset_name,
                               split_idx=split_idx, sd_name=sd_name)
    return results


# Write exception to a file whose name contains a timestamp.
def error_callback(e: Exception) -> None:
    file_name = 'error_' + time.strftime('%H_%M_%S', time.localtime()) + '.txt'
    with open(file_name, mode="w") as file:
        file.write(str(e))


# Main routine: Run complete experimental pipeline. To this end, read datasets from "data_dir",
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
    context = multiprocessing.get_context(method='spawn')
    process_pool = context.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_experimental_task, kwds=task,
                                        callback=lambda x: progress_bar.update(),
                                        error_callback=error_callback)
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
