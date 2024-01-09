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
    alt_evaluation_metrics = ['alt.hamming', 'alt.jaccard']

    # Compute further evaluation metrics:
    results['time_fit_opt_diff'] = results['fitting_time'] - results['optimization_time']
    results['wracc_train_test_diff'] = results['train_wracc'] - results['test_wracc']

    # Define constants for filtering results:
    max_k = 'no'  # placeholder value for unlimited cardinality (else not appearing in groupby())
    results['param.k'].fillna(max_k, inplace=True)
    max_timeout = results['param.timeout'].max()
    min_tau_abs = results['param.tau_abs'].min()  # could also be any other unique value of tau_abs

    print('\n---- Default analysis (max timeout, max cardinality, no alternatives) ----')

    eval_results = results[results['param.timeout'].isin([float('nan'), max_timeout]) &
                           (results['param.k'] == max_k) &
                           results['alt.number'].isin([float('nan'), 0]) &
                           results['param.tau_abs'].isin([float('nan'), min_tau_abs])]
    no_timeout_datasets = eval_results.groupby('dataset_name')['optimization_status'].agg(
        lambda x: (x != 'unknown').all())  # returns Series with bool values and DS names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()

    print('\nHow are the mean values of evaluation metrics distributed (all datasets)?')
    print(eval_results.groupby('sd_name')[evaluation_metrics].mean().round(3))

    print('\nHow are the mean values of evaluation metrics distributed (datasets without timeout',
          'in exact optimization)?')
    print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
        'sd_name')[evaluation_metrics].mean().round(3))

    print('\nHow is the difference "train - test" in WRAcc distributed?')
    print(eval_results.groupby('sd_name')['wracc_train_test_diff'].describe().transpose().round(3))

    print('\nHow is the difference "SMT - Beam" in the values of evaluation metrics distributed',
          '(all datasets)?')
    print_results = eval_results.pivot(index=['dataset_name', 'split_idx'], columns='sd_name',
                                       values=evaluation_metrics).reset_index()
    for metric in evaluation_metrics:
        print_results[(metric, 'diff')] = (print_results[(metric, 'SMT')] -
                                           print_results[(metric, 'Beam')])
    print_results = print_results.loc[:, (slice(None), ['', 'diff'])]  # keep "diff" & ID cols
    print_results = print_results.droplevel(level='sd_name', axis='columns')
    print(print_results[evaluation_metrics].describe().round(3))

    print('\nHow is the difference "SMT - Beam" in the values of evaluation metrics distributed',
          '(datasets without timeout in exact optimization)?')
    print(print_results.loc[print_results['dataset_name'].isin(no_timeout_datasets),
                            evaluation_metrics].describe().round(3))

    print('\nHow is the difference "entire fitting - only optimization" in runtime distributed?')
    print(eval_results.groupby('sd_name')['time_fit_opt_diff'].describe().transpose().round(2))

    print('\nHow is the runtime Spearman-correlated to dataset size?')
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)
    dataset_overview = dataset_overview[['dataset', 'n_instances', 'n_features']]
    print_results = dataset_overview.rename(columns={'n_instances': 'm', 'n_features': 'n'})
    print_results['n*m'] = print_results['n'] * print_results['m']
    print_results['sum_unique_values'] = print_results['dataset'].apply(
        lambda dataset_name: data_handling.load_dataset(
            dataset_name=dataset_name, directory=data_dir)[0].nunique().sum())  # 0 = "X", 1 = "y"
    print_results = print_results.merge(
        eval_results[['dataset_name', 'sd_name', 'optimization_time', 'fitting_time']].rename(
            columns={'dataset_name': 'dataset'})).drop(columns='dataset')
    print(print_results.groupby('sd_name').corr(method='spearman').round(2))

    print('\n---- Timeout analysis ----')

    print('\nHow is the number of finished SMT tasks distributed over timeouts and cardinality?')
    print_results = results.loc[(results['sd_name'] == 'SMT') &
                                results['alt.number'].isin([float('nan'), 0]) &
                                results['param.tau_abs'].isin([float('nan'), min_tau_abs]),
                                ['param.k', 'param.timeout', 'optimization_status']].copy()
    print_results = print_results.groupby(['param.k', 'param.timeout'])['optimization_status'].agg(
        lambda x: (x == 'sat').sum() / len(x)).rename('finished').reset_index()
    print(print_results.pivot(index='param.k', columns='param.timeout').applymap('{:.1%}'.format))

    eval_results = results[(results['sd_name'] == 'SMT') & (results['param.k'] == max_k) &
                           results['alt.number'].isin([float('nan'), 0]) &
                           results['param.tau_abs'].isin([float('nan'), min_tau_abs])]
    all_timeout_datasets = eval_results.groupby('dataset_name')['optimization_status'].agg(
        lambda x: (x == 'unknown').all())  # returns Series with bool values and DS names as index
    all_timeout_datasets = all_timeout_datasets[all_timeout_datasets].index.to_list()

    print('\nHow is the mean value of evaluation metrics distributed over timeouts (with maximum',
          'cardinality and all datasets)?')
    print(eval_results.groupby('param.timeout')[evaluation_metrics].mean().round(3))

    print('\nHow is the mean value of evaluation metrics distributed over timeouts (with maximum',
          'cardinality and timeout-only datasets)?')
    print(eval_results[eval_results['dataset_name'].isin(all_timeout_datasets)].groupby(
        'param.timeout')[evaluation_metrics].mean().round(3))

    print('\nHow is the difference "train - test" in WRAcc distributed over timeouts (with',
          'maximum cardinality and all datasets)?')
    print(eval_results.groupby('param.timeout')['wracc_train_test_diff'].describe().round(3))

    print('\nHow is the difference "train - test" in WRAcc distributed over timeouts (with',
          'maximum cardinality and timeout-only datasets)?')
    print(eval_results[eval_results['dataset_name'].isin(all_timeout_datasets)].groupby(
        'param.timeout')['wracc_train_test_diff'].describe().round(3))

    print('\n---- Cardinality analysis (max timeout) ----')

    eval_results = results[results['param.timeout'].isin([float('nan'), max_timeout]) &
                           results['alt.number'].isin([float('nan'), 0]) &
                           results['param.tau_abs'].isin([float('nan'), min_tau_abs])]
    no_timeout_datasets = eval_results.groupby('dataset_name')['optimization_status'].agg(
        lambda x: (x != 'unknown').all())  # returns Series with bool values and DS names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()

    print('\nHow are the mean values of evaluation metrics distributed over cardinality "k"',
          '(all datasets)?')
    for metric in evaluation_metrics:
        print(eval_results.groupby(['sd_name', 'param.k'])[metric].mean().reset_index().pivot(
            index='param.k', columns='sd_name').round(3))

    print('\nHow are the mean values of evaluation metrics distributed over cardinality "k"',
          '(datasets without timeout in exact optimization)?')
    for metric in evaluation_metrics:
        print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
            ['sd_name', 'param.k'])[metric].mean().reset_index().pivot(
                index='param.k', columns='sd_name').round(3))

    print('\nHow is the mean difference "train - test" in WRAcc distributed over cardinality "k"?')
    print(eval_results.groupby(['sd_name', 'param.k'])['wracc_train_test_diff'].mean(
        ).reset_index().pivot(index='param.k', columns='sd_name').round(3))

    print('\nHow is the difference "SMT - Beam" in the values of evaluation metrics distributed',
          'over cardinality "k" (all datasets)?')
    print_results = eval_results.pivot(index=['dataset_name', 'split_idx', 'param.k'],
                                       columns='sd_name', values=evaluation_metrics).reset_index()
    for metric in evaluation_metrics:
        print_results[(metric, 'diff')] = (print_results[(metric, 'SMT')] -
                                           print_results[(metric, 'Beam')])
    print_results = print_results.loc[:, (slice(None), ['', 'diff'])]  # keep "diff" & ID cols
    print_results = print_results.droplevel(level='sd_name', axis='columns')
    for metric in evaluation_metrics:
        print(f'Metric: {metric}')
        print(print_results.groupby('param.k')[metric].describe().round(3))

    print('\nHow is the difference "SMT - Beam" in the values of evaluation metrics distributed',
          'over cardinality "k" (datasets without timeout in exact optimization)?')
    for metric in evaluation_metrics:
        print(f'Metric: {metric}')
        print(print_results[print_results['dataset_name'].isin(no_timeout_datasets)].groupby(
            'param.k')[metric].describe().round(3))

    print('\n---- Alternatives analysis (fixed cardinality, max timeout) ----')

    eval_results = results[results['alt.number'].notna()]
    no_timeout_datasets = eval_results.groupby('dataset_name')['optimization_status'].agg(
        lambda x: (x != 'unknown').all())  # returns Series with bool values and DS names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()

    print('\nHow is the number of finished SMT tasks distributed over the number of alternative',
          'and the dissimilarity threshold?')
    print(eval_results.groupby(['alt.number', 'param.tau_abs'])['optimization_status'].agg(
        lambda x: (x == 'sat').sum() / len(x)).rename('finished').reset_index().pivot(
            index='alt.number', columns='param.tau_abs').applymap('{:.1%}'.format))

    print('\nHow are the mean values of evaluation metrics distributed over the number of',
          'alternative and the dissimilarity threshold (all datasets)?')
    for metric in evaluation_metrics + ['wracc_train_test_diff'] + alt_evaluation_metrics:
        print(eval_results.groupby(['alt.number', 'param.tau_abs'])[metric].mean().reset_index(
            ).pivot(index='alt.number', columns='param.tau_abs').round(3))

    print('\nHow are the mean values of evaluation metrics distributed over the number of',
          'alternative and the dissimilarity threshold (datasets without timeout)?')
    for metric in evaluation_metrics + ['wracc_train_test_diff'] + alt_evaluation_metrics:
        print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
            ['alt.number', 'param.tau_abs'])[metric].mean().reset_index().pivot(
                index='alt.number', columns='param.tau_abs').round(3))


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
