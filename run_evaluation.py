"""Run evaluation

Script to compute summary statistics and create plots for the paper. Should be run after the
experimental pipeline, as this script requires the pipeline's outputs as inputs.

Usage: python -m run_evaluation --help
"""


import argparse
import ast
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data_handling


plt.rcParams['font.family'] = 'Arial'


# Sum the number of unique values over all features in a dataset.
def sum_unique_values(dataset_name: str, data_dir: pathlib.Path) -> int:
    X, _ = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    return X.nunique().sum()


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
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)

    # Make feature sets proper lists (converting lbs/ubs would be harder, since they contain inf):
    results['selected_feature_idxs'] = results['selected_feature_idxs'].apply(ast.literal_eval)
    assert (results['param.k'].isna() |
            (results['selected_feature_idxs'].apply(len) <= results['param.k'])).all()

    # Define column list for evaluation:
    evaluation_metrics = ['optimization_time', 'fitting_time', 'train_nwracc', 'test_nwracc']
    alt_evaluation_metrics = ['alt.hamming', 'alt.jaccard']
    sd_name_plot_order = ['SMT', 'Beam', 'BI', 'PRIM', 'MORB', 'Random']

    # Compute further evaluation metrics:
    results['time_fit_opt_diff'] = results['fitting_time'] - results['optimization_time']
    results['nwracc_train_test_diff'] = results['train_nwracc'] - results['test_nwracc']

    # Define constants for filtering results:
    max_k = 'no'  # placeholder value for unlimited cardinality (else not appearing in groupby())
    results['param.k'].fillna(max_k, inplace=True)
    max_timeout = results['param.timeout'].max()
    min_tau_abs = results['param.tau_abs'].min()  # could also be any other unique value of tau_abs

    print('\n-------- Experimental Design --------')

    print('\n------ Datasets ------')

    print('\n## Table 1: Dataset overview ##\n')
    print_results = dataset_overview[['dataset', 'n_instances', 'n_features']].rename(
        columns={'dataset': 'Dataset', 'n_instances': '$m$', 'n_features': '$n$'})
    print_results['max-k'] = print_results['Dataset'].apply(
        lambda dataset_name: (results.loc[
            (results['dataset_name'] == dataset_name) & (results['sd_name'] == 'SMT') &
            (results['param.timeout'] == max_timeout) & (results['param.k'] == max_k) &
            results['alt.number'].isna(), 'optimization_status'] != 'sat').any())
    print_results['any-k'] = print_results['Dataset'].apply(
        lambda dataset_name: (results.loc[
            (results['dataset_name'] == dataset_name) & (results['sd_name'] == 'SMT') &
            (results['param.timeout'] == max_timeout) &
            results['alt.number'].isin([float('nan'), 0]) &
            results['param.tau_abs'].isin([float('nan'), min_tau_abs]),
            'optimization_status'] != 'sat').any())
    print_results.replace({False: 'No', True: 'Yes'}, inplace=True)
    print_results['Dataset'] = print_results['Dataset'].str.replace('GAMETES', 'G')
    print_results['Dataset'] = print_results['Dataset'].str.replace('_Epistasis', 'E')
    print_results['Dataset'] = print_results['Dataset'].str.replace('_Heterogeneity', 'H')
    print_results.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(print_results.style.format(escape='latex', precision=2).hide(axis='index').to_latex(
        hrules=True))

    print('\n-------- Evaluation --------')

    print('\n------ Experimental scenario 1: Unconstrained subgroup discovery',
          '(max timeout, max cardinality, no alternatives) ------')

    eval_results = results[results['param.timeout'].isin([float('nan'), max_timeout]) &
                           (results['param.k'] == max_k) &
                           results['alt.number'].isin([float('nan'), 0]) &
                           results['param.tau_abs'].isin([float('nan'), min_tau_abs])]
    all_datasets = eval_results['dataset_name'].unique()
    no_timeout_datasets = eval_results[eval_results['sd_name'] == 'SMT'].groupby('dataset_name')[
        'optimization_status'].agg(lambda x: (x == 'sat').all())  # bool Series with names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()

    print('\nHow are the mean values of evaluation metrics distributed (all datasets)?')
    print(eval_results.groupby('sd_name')[evaluation_metrics].mean().round(3))

    print('\nHow are the mean values of evaluation metrics distributed (datasets without timeout',
          'in exact optimization)?')
    print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
        'sd_name')[evaluation_metrics].mean().round(3))

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

    print('\n-- Subgroup quality --')

    # Figures 1a, 1b: Subgroup quality by subgroup-discovery method
    plot_results = eval_results[['dataset_name', 'sd_name', 'train_nwracc', 'test_nwracc']]
    plot_results = plot_results.melt(id_vars=['sd_name', 'dataset_name'], value_name='nWRAcc',
                                     value_vars=['train_nwracc', 'test_nwracc'], var_name='Split')
    plot_results['Split'] = plot_results['Split'].str.replace('_nwracc', '')
    for (dataset_list, selection_name) in [(all_datasets, 'all-datasets'),
                                           (no_timeout_datasets, 'no-timeout-datasets')]:
        plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = 18
        sns.boxplot(data=plot_results[plot_results['dataset_name'].isin(dataset_list)],
                    x='sd_name', y='nWRAcc', hue='Split', palette='Set2', order=sd_name_plot_order)
        plt.xlabel('Subgroup-discovery method')
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.ylim(-0.35, 1.05)
        plt.yticks(np.arange(start=-0.2, stop=1.2, step=0.2))
        leg = plt.legend(title='Split', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.3), columnspacing=1, framealpha=0, ncols=2)
        leg.get_title().set_position((-127, -25))
        plt.tight_layout()
        plt.savefig(plot_dir / f'csd-methods-unconstrained-nwracc-{selection_name}.pdf')

    print('\nHow is the difference "train - test" in nWRAcc distributed?')
    print(eval_results.groupby('sd_name')['nwracc_train_test_diff'].describe().transpose().round(3))

    print('\n-- Runtime --')

    print('\n## Tables 2a, 2b: Aggregated runtime by subgroup-discovery-method ##\n')
    for (dataset_list, selection_name) in [(all_datasets, 'all-datasets'),
                                           (no_timeout_datasets, 'no-timeout-datasets')]:
        print('Dataset selection:', selection_name)
        print_results = eval_results[eval_results['dataset_name'].isin(dataset_list)]
        print_results = print_results.groupby('sd_name')['fitting_time'].agg(
            ['mean', 'std', 'median'])
        print_results.index.rename('Runtime', inplace=True)
        print_results = print_results.reindex(sd_name_plot_order)
        print_results.rename(columns={'mean': 'Mean', 'std': 'Standard dev.', 'median': 'Median'},
                             inplace=True)
        print(print_results.transpose().style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\nHow is the difference "entire fitting - only optimization" in runtime distributed?')
    print(eval_results.groupby('sd_name')['time_fit_opt_diff'].describe().transpose().round(2))

    print('\n## Table 3: Correlation of runtime by subgroup-discovery method',
          '(datasets without timeout in exact optimization)##\n')
    print_results = dataset_overview[['dataset', 'n_instances', 'n_features']].rename(
        columns={'dataset': 'dataset_name', 'n_instances': '$m$', 'n_features': '$n$'})
    print_results['$m \\cdot n$'] = print_results['$m$'] * print_results['$n$']
    print_results['$\\Sigma n^u$'] = print_results['dataset_name'].apply(
        sum_unique_values, data_dir=data_dir)
    print_results = print_results.merge(
        eval_results.loc[eval_results['dataset_name'].isin(no_timeout_datasets),
                         ['dataset_name', 'sd_name', 'fitting_time']]).drop(columns='dataset_name')
    print_results = print_results.groupby('sd_name').corr(method='spearman')
    print_results = print_results['fitting_time'].reset_index()
    print_results = print_results.pivot(index='sd_name', columns='level_1', values='fitting_time')
    print_results.index.name = None  # left-over of pivot()
    print_results = print_results.reindex(sd_name_plot_order)
    print_results.columns.name = 'Method'
    print_results = print_results.drop(columns='fitting_time')
    print(print_results.style.format('{:.2f}'.format).to_latex(hrules=True))

    print('\n------ Experimental scenario 2: Solver timeouts (no alternatives) ------')

    print('\nHow is the number of finished SMT tasks distributed over timeouts and cardinality?')
    print_results = results.loc[(results['sd_name'] == 'SMT') &
                                results['alt.number'].isin([float('nan'), 0]) &
                                results['param.tau_abs'].isin([float('nan'), min_tau_abs]),
                                ['param.k', 'param.timeout', 'optimization_status']].copy()
    print_results = print_results.groupby(['param.k', 'param.timeout'])['optimization_status'].agg(
        lambda x: (x == 'sat').sum() / len(x)).rename('finished').reset_index()
    print(print_results.pivot(index='param.timeout', columns='param.k').applymap('{:.1%}'.format))

    eval_results = results[(results['sd_name'] == 'SMT') & (results['param.k'] == max_k) &
                           results['alt.number'].isin([float('nan'), 0]) &
                           results['param.tau_abs'].isin([float('nan'), min_tau_abs])]
    all_timeout_datasets = eval_results.groupby('dataset_name')['optimization_status'].agg(
        lambda x: (x != 'sat').all())  # returns Series with bool values and DS names as index
    all_timeout_datasets = all_timeout_datasets[all_timeout_datasets].index.to_list()

    print('\nHow is the mean value of evaluation metrics distributed over timeouts (with maximum',
          'cardinality and all datasets)?')
    print(eval_results.groupby('param.timeout')[evaluation_metrics].mean().round(3))

    print('\nHow is the mean value of evaluation metrics distributed over timeouts (with maximum',
          'cardinality and timeout-only datasets)?')
    print(eval_results[eval_results['dataset_name'].isin(all_timeout_datasets)].groupby(
        'param.timeout')[evaluation_metrics].mean().round(3))

    print('\nHow is the difference "train - test" in nWRAcc distributed over timeouts (with',
          'maximum cardinality and all datasets)?')
    print(eval_results.groupby('param.timeout')['nwracc_train_test_diff'].describe().round(3))

    print('\nHow is the difference "train - test" in nWRAcc distributed over timeouts (with',
          'maximum cardinality and timeout-only datasets)?')
    print(eval_results[eval_results['dataset_name'].isin(all_timeout_datasets)].groupby(
        'param.timeout')['nwracc_train_test_diff'].describe().round(3))

    print('\n------ Experimental scenario 3: Feature-cardinality constraints',
          '(max timeout, no alternatives) ------')

    eval_results = results[results['param.timeout'].isin([float('nan'), max_timeout]) &
                           results['alt.number'].isin([float('nan'), 0]) &
                           results['param.tau_abs'].isin([float('nan'), min_tau_abs])]
    no_timeout_datasets = eval_results[eval_results['sd_name'] == 'SMT'].groupby('dataset_name')[
        'optimization_status'].agg(lambda x: (x == 'sat').all())  # bool Series with names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()

    print('\nHow does the number of actually selected features differ from the prescribed "k"?')
    print(pd.crosstab(
        eval_results.loc[eval_results['param.k'] != max_k, 'param.k'],
        eval_results.loc[eval_results['param.k'] != max_k, 'selected_feature_idxs'].apply(
            len).rename('actually selected')))

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

    print('\nHow is the mean difference "train - test" in nWRAcc distributed over cardinality "k"',
          '(all datasets)?')
    print(eval_results.groupby(['sd_name', 'param.k'])['nwracc_train_test_diff'].mean(
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

    print('\n------ Experimental scenario 4: Alternative subgroup descriptions',
          '(max timeout, fixed cardinality) ------')

    eval_results = results[results['alt.number'].notna()]
    no_timeout_datasets = eval_results[eval_results['sd_name'] == 'SMT'].groupby('dataset_name')[
        'optimization_status'].agg(lambda x: (x == 'sat').all())  # bool Series with names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()

    print('\nHow is the number of finished SMT tasks distributed over the number of alternative',
          'and the dissimilarity threshold?')
    print(eval_results[eval_results['sd_name'] == 'SMT'].groupby(['alt.number', 'param.tau_abs'])[
        'optimization_status'].agg(lambda x: (x == 'sat').sum() / len(x)).rename('').reset_index(
            ).pivot(index='alt.number', columns='param.tau_abs').applymap('{:.1%}'.format))

    print('\nHow are the mean values of evaluation metrics distributed over the number of',
          'alternative and the dissimilarity threshold (all datasets)?')
    for metric in evaluation_metrics + ['nwracc_train_test_diff'] + alt_evaluation_metrics:
        print(eval_results.groupby(['sd_name', 'alt.number', 'param.tau_abs'])[metric].mean(
            ).reset_index().pivot(index=['sd_name', 'alt.number'], columns='param.tau_abs').round(3))

    print('\nHow are the mean values of evaluation metrics distributed over the number of',
          'alternative and the dissimilarity threshold (datasets without timeout)?')
    for metric in evaluation_metrics + ['nwracc_train_test_diff'] + alt_evaluation_metrics:
        print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
            ['sd_name', 'alt.number', 'param.tau_abs'])[metric].mean().reset_index().pivot(
                index=['sd_name', 'alt.number'], columns='param.tau_abs').round(3))

    print('\nHow are the mean values of evaluation metrics (shifted to [0, 1] and max-normalized',
          'with quality of original subgroup) distributed over the number of alternative and the',
          'dissimilarity threshold (all datasets)?')
    norm_metrics = ['train_nwracc', 'test_nwracc']
    norm_group_cols = ['dataset_name', 'split_idx', 'sd_name', 'param.tau_abs']
    norm_results = eval_results[norm_group_cols + ['alt.number'] + norm_metrics].copy()
    norm_results[norm_metrics] = (norm_results[norm_metrics] + 1) / 2  # from [-1, 1] to [0, 1]
    assert norm_results.groupby(norm_group_cols)['alt.number'].is_monotonic_increasing.all()
    norm_results[norm_metrics] = norm_results.groupby(norm_group_cols)[norm_metrics].transform(
        lambda x: x / x.iloc[0])  # original subgroup is 1st row in each group (see assertion)
    for metric in norm_metrics:
        print(norm_results.groupby(['sd_name', 'alt.number', 'param.tau_abs'])[metric].mean(
            ).reset_index().pivot(index=['sd_name', 'alt.number'], columns='param.tau_abs').round(3))


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
