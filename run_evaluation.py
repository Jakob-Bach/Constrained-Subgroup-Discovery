"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
"""


import argparse
import pathlib

import matplotlib.pyplot as plt

import data_handling


plt.rcParams['font.family'] = 'Arial'


# Main-routine: Run complete evaluation pipeline. To this end, read results from the "results_dir"
# and some dataset information from "data_dir". Save plots to the "plot_dir". Print some statistics
# to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('The results directory does not exist.')
    if not plot_dir.is_dir():
        print('The plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if any(plot_dir.glob('*.pdf')):
        print('The plot directory is not empty. Files might be overwritten but not deleted.')

    results = data_handling.load_results(directory=results_dir)

    # Define column list for evaluation:
    evaluation_metrics = ['optimization_time', 'fitting_time', 'train_wracc', 'test_wracc']

    print('\nHow are the optimization statuses distributed?')
    print(results.groupby('sd_name')['optimization_status'].value_counts())

    print('\nHow are the values of evaluation metrics distributed?')
    print(results.groupby('sd_name')[evaluation_metrics].describe().transpose().round(2))

    print('\nHow is the difference "train - test" in WRAcc distributed?')
    print_results = results[['sd_name', 'dataset_name', 'split_idx']].copy()
    print_results['diff'] = results['train_wracc'] - results['test_wracc']
    print(print_results.groupby('sd_name')['diff'].describe().transpose().round(2))

    print('\nHow is the difference "entire fitting - only optimization" in runtime distributed?')
    print_results = results[['sd_name', 'dataset_name', 'split_idx']].copy()
    print_results['diff'] = results['fitting_time'] - results['optimization_time']
    print(print_results.groupby('sd_name')['diff'].describe().transpose().round(2))

    print('\nHow is the difference "MIP - SMT" in the values of evaluation metrics distributed?')
    print_results = results.pivot(index=['dataset_name', 'split_idx'], columns='sd_name',
                                  values=evaluation_metrics).reset_index()
    for metric in evaluation_metrics:
        print_results[(metric, 'diff')] = (print_results[(metric, 'MIP')] -
                                           print_results[(metric, 'SMT')])
    print_results = print_results.loc[:, (slice(None), ['', 'diff'])]  # keep "diff" & ID cols
    print_results = print_results.droplevel(level='sd_name', axis='columns')
    print(print_results[evaluation_metrics].describe().round(2))

    print('\nHow is the runtime Spearman-correlated to dataset size?')
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    print_results = dataset_overview.rename(columns={'n_instances': 'm', 'n_features': 'n'})
    print_results['n*m'] = print_results['n'] * print_results['m']
    print_results = print_results.merge(
        results[['dataset_name', 'sd_name', 'optimization_time', 'fitting_time']].rename(
            columns={'dataset_name': 'dataset'})).drop(columns='dataset')
    print(print_results.groupby('sd_name').corr(method='spearman').round(2))


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates paper\'s plots and prints statistics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/datasets/', dest='data_dir',
                        help='Directory with prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.\n')
    evaluate(**vars(parser.parse_args()))
    print('\nEvaluation finished. Plots created and saved.')
