"""Run arXiv evaluation

Script to compute summary statistics and create plots + tables for the arXiv version of the paper.
Should be run after the experimental pipeline, as this evaluation script requires the pipeline's
outputs as inputs.

Usage: python -m run_evaluation_arxiv --help
"""


import argparse
import ast
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets

import csd
import data_handling
from run_experiments import SD4PY_METHODS, SD4PY_TIMEOUT_DATASETS


plt.rcParams['font.family'] = 'Arial'
DEFAULT_COL_PALETTE = 'YlGnBu'


# Sum the number of unique values over all features in a dataset.
def sum_unique_values(dataset_name: str, data_dir: pathlib.Path) -> int:
    X, _ = data_handling.load_dataset(dataset_name=dataset_name, directory=data_dir)
    return X.nunique().sum()


# Main routine: Run complete evaluation pipeline. To this end, read results from the "results_dir"
# and some dataset information from "data_dir". Save plots to the "plot_dir". Print some statistics
# and LaTeX-ready tables to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('The results directory does not exist.')
    if not plot_dir.is_dir():
        print('The plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if any(plot_dir.iterdir()):
        print('The plot directory is not empty. Files might be overwritten but not deleted.')

    results = data_handling.load_results(directory=results_dir)
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)

    # For some datasets, we did not run the exhaustive SD4Py methods in the unconstrained scenario
    # (would have taken too long); fill these missing rows with results from max fixed "k" (k=5):
    fill_results = results[(results['dataset_name'].isin(SD4PY_TIMEOUT_DATASETS)) &
                           (results['sd_name'].isin(SD4PY_METHODS)) &
                           (results['param.k'] == results['param.k'].max())].copy()
    fill_results['param.k'] = float('nan')  # value of "k" for unconstrained scenario
    results = pd.concat([results, fill_results], ignore_index=True)

    # Make feature sets proper lists (converting lbs/ubs would be harder, since they contain inf):
    results['selected_feature_idxs'] = results['selected_feature_idxs'].apply(ast.literal_eval)
    assert (results['param.k'].isna() |
            (results['selected_feature_idxs'].apply(len) <= results['param.k'])).all()

    # Compute further evaluation metrics and define column lists for evaluation:
    results['time_fit_opt_diff'] = results['fitting_time'] - results['optimization_time']
    results['nwracc_train_test_diff'] = results['train_nwracc'] - results['test_nwracc']
    evaluation_metrics = ['fitting_time', 'train_nwracc', 'test_nwracc', 'nwracc_train_test_diff']
    alt_evaluation_metrics = ['alt.hamming', 'alt.jaccard']
    sd_name_plot_order = ['SMT', 'Beam', 'BI', 'BSD', 'MORS', 'PRIM', 'Random', 'SD-Map']

    # Define constants for filtering results:
    int_na_columns = ['param.k', 'param.timeout', 'param.tau_abs', 'alt.number']
    results[int_na_columns] = results[int_na_columns].astype('Int64')  # int with NAs
    max_k = 'no'  # placeholder value for unlimited cardinality (else not appearing in groupby())
    results['param.k'] = results['param.k'].astype('object').fillna(max_k)
    max_timeout = results['param.timeout'].max()
    min_tau_abs = results['param.tau_abs'].min()  # could also be any other unique value of tau_abs

    print('\n-------- 1 Introduction --------')

    print('\n-- Motivation --')

    X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
    X = X[['petal length (cm)', 'petal width (cm)']]
    X.columns = [f'Feature_{j + 1}' for j in range(X.shape[1])]
    y = (y == 1).astype(int).rename('Target')
    plot_data = pd.concat((X, y.astype(str)), axis='columns')

    model = csd.MORSSubgroupDiscoverer(k=2)
    model.fit(X=X, y=y)

    print('\nWhat are the lower bounds of the exemplary subgroup?')
    print(model.get_box_lbs())

    print('\nWhat are the upper bounds of the exemplary subgroup?')
    print(model.get_box_ubs())

    # Figure 1: Exemplary subgroup description
    j_1, j_2 = model.get_selected_feature_idxs()
    plt.figure(figsize=(8, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x=plot_data.columns[j_1], y=plot_data.columns[j_2], hue='Target',
                    style='Target', data=plot_data, palette=DEFAULT_COL_PALETTE)
    plt.vlines(x=(model.get_box_lbs()[j_1], model.get_box_ubs()[j_1]),
               ymin=model.get_box_lbs()[j_2], ymax=model.get_box_ubs()[j_2],
               colors=sns.color_palette(DEFAULT_COL_PALETTE, 2)[1])
    plt.hlines(y=(model.get_box_lbs()[j_2], model.get_box_ubs()[j_2]),
               xmin=model.get_box_lbs()[j_1], xmax=model.get_box_ubs()[j_1],
               colors=sns.color_palette(DEFAULT_COL_PALETTE, 2)[1])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(plot_dir / 'csd-exemplary-subgroup.pdf')

    print('\n-------- 5 Experimental Design --------')

    print('\n------ 5.5 Datasets ------')

    print('\n## Table 1: Dataset overview ##\n')
    print_results = dataset_overview[['dataset', 'n_instances', 'n_features']].rename(
        columns={'dataset': 'Dataset', 'n_instances': '$m$', 'n_features': '$n$'})
    print_results['timeouts-for-max-k'] = print_results['Dataset'].apply(
        lambda dataset_name: (results.loc[
            (results['dataset_name'] == dataset_name) & (results['sd_name'] == 'SMT') &
            (results['param.timeout'] == max_timeout) & (results['param.k'] == max_k) &
            results['alt.number'].isna() & results['param.tau_abs'].isna(),
            'optimization_status'] != 'sat').any())
    print_results['timeouts-for-any-k'] = print_results['Dataset'].apply(
        lambda dataset_name: (results.loc[
            (results['dataset_name'] == dataset_name) & (results['sd_name'] == 'SMT') &
            (results['param.timeout'] == max_timeout) &
            results['alt.number'].isin([pd.NA, 0]) &
            results['param.tau_abs'].isin([pd.NA, min_tau_abs]),
            'optimization_status'] != 'sat').any())
    print_results.replace({False: 'No', True: 'Yes'}, inplace=True)
    print_results['Dataset'] = print_results['Dataset'].str.replace('GAMETES', 'G')
    print_results['Dataset'] = print_results['Dataset'].str.replace('_Epistasis', 'E')
    print_results['Dataset'] = print_results['Dataset'].str.replace('_Heterogeneity', 'H')
    print_results.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(print_results.style.format(escape='latex', precision=2).hide(axis='index').to_latex(
        hrules=True))

    print('\n-------- 6 Evaluation --------')

    print('\n------ 6.1 Unconstrained Subgroup Discovery ------')
    # max timeout, max cardinality, no alternatives

    eval_results = results[results['param.timeout'].isin([pd.NA, max_timeout]) &
                           (results['param.k'] == max_k) &
                           results['alt.number'].isin([pd.NA, 0]) &
                           results['param.tau_abs'].isin([pd.NA, min_tau_abs])]
    all_datasets = eval_results['dataset_name'].unique()
    no_timeout_datasets = eval_results[eval_results['sd_name'] == 'SMT'].groupby('dataset_name')[
        'optimization_status'].agg(lambda x: (x == 'sat').all())  # bool Series with names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()
    print('\nNumber of datasets without solver timeouts:', len(no_timeout_datasets))

    print('\nWhat is the mean value of evaluation metrics for different subgroup-discovery',
          'methods (for all datasets)?')
    print(eval_results.groupby('sd_name')[evaluation_metrics].mean().round(3))

    print('\nWhat is the mean value of evaluation metrics for different subgroup-discovery',
          'methods (for datasets without timeouts in SMT optimization)?')
    print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
        'sd_name')[evaluation_metrics].mean().round(3))

    print('\nHow is the difference "SMT - Beam" distributed for different evaluation metrics',
          '(for all datasets)?')
    print_results = eval_results.pivot(index=['dataset_name', 'split_idx'], columns='sd_name',
                                       values=evaluation_metrics).reset_index()
    for metric in evaluation_metrics:
        print_results[(metric, 'diff')] = (print_results[(metric, 'SMT')] -
                                           print_results[(metric, 'Beam')])
    print_results = print_results.loc[:, (slice(None), ['', 'diff'])]  # keep "diff" & ID cols
    print_results = print_results.droplevel(level='sd_name', axis='columns')
    print(print_results[evaluation_metrics].describe().round(3))

    print('\nHow is the difference "SMT - Beam" distributed for different evaluation metrics',
          '(for datasets without timeouts in SMT optimization)?')
    print(print_results.loc[print_results['dataset_name'].isin(no_timeout_datasets),
                            evaluation_metrics].describe().round(3))

    print('\n-- Subgroup quality --')

    # Figures 2a, 2b: Subgroup quality by subgroup-discovery method (subfigures: timeouts y/n)
    plot_results = eval_results[['dataset_name', 'sd_name', 'train_nwracc', 'test_nwracc']]
    plot_results = plot_results.melt(id_vars=['sd_name', 'dataset_name'], value_name='nWRAcc',
                                     value_vars=['train_nwracc', 'test_nwracc'], var_name='Split')
    plot_results['Split'] = plot_results['Split'].str.replace('_nwracc', '')
    for (dataset_list, selection_name) in [(all_datasets, 'all-datasets'),
                                           (no_timeout_datasets, 'no-timeout-datasets')]:
        plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = 18
        sns.boxplot(data=plot_results[plot_results['dataset_name'].isin(dataset_list)],
                    x='sd_name', y='nWRAcc', hue='Split', palette=DEFAULT_COL_PALETTE,
                    order=sd_name_plot_order)
        plt.xlabel('Subgroup-discovery method')
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.ylim(-0.35, 1.05)
        plt.yticks(np.arange(start=-0.2, stop=1.2, step=0.2))
        leg = plt.legend(title='Split', edgecolor='white', loc='upper left',
                         bbox_to_anchor=(0, -0.3), columnspacing=1, framealpha=0, ncols=2)
        leg.get_title().set_position((-127, -25))
        plt.tight_layout()
        plt.savefig(plot_dir / f'csd-unconstrained-nwracc-{selection_name}.pdf')

    print('\n-- Runtime --')

    print('\n## Tables 2a, 2b: Aggregated runtime by aggregate and subgroup-discovery method',
          '(subtables: timeouts y/n) ##\n')
    for (dataset_list, selection_name) in [(all_datasets, 'all-datasets'),
                                           (no_timeout_datasets, 'no-timeout-datasets')]:
        print('Dataset selection:', selection_name, '\n')
        print_results = eval_results[eval_results['dataset_name'].isin(dataset_list)]
        print_results = print_results.groupby('sd_name')['fitting_time'].agg(
            ['mean', 'std', 'median'])
        print_results.index.name = None  # would be an unnecessary row in table
        print_results.columns.name = 'Method'
        print_results.rename(columns={'mean': 'Mean', 'std': 'Standard dev.', 'median': 'Median'},
                             inplace=True)
        print(print_results.style.format('{:.2f}~s'.format).to_latex(hrules=True))

    print('\nHow is the difference "entire fitting - only optimization" in runtime distributed',
          'for different subgroup-discovery methods?')
    print(eval_results.groupby('sd_name')['time_fit_opt_diff'].describe().transpose().round(2))

    print('\n## Table 3: Correlation of runtime by subgroup-discovery method and dataset-size',
          'metric (for datasets without timeouts in SMT optimization) ##\n')
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
    print_results.index.name = None  # left-over of pivot(), would be an unnecessary row in table
    print_results.columns.name = 'Method'
    print_results = print_results.drop(columns='fitting_time')  # self-correlation is boring
    print(print_results.style.format('{:.2f}'.format).to_latex(hrules=True))

    print('\n------ 6.2 Solver Timeouts ------')
    # no alternatives

    print('\n-- Finished tasks --')

    print('\nWhat is the frequency of finished SMT tasks for different solver timeouts and',
          'feature-cardinality thresholds?')
    eval_results = results.loc[(results['sd_name'] == 'SMT') &
                               results['alt.number'].isin([pd.NA, 0]) &
                               results['param.tau_abs'].isin([pd.NA, min_tau_abs])]
    print_results = eval_results.groupby(['param.k', 'param.timeout'])['optimization_status'].agg(
        lambda x: (x == 'sat').sum() / len(x)).rename('finished').reset_index()
    print(print_results.pivot(index='param.timeout', columns='param.k').applymap('{:.1%}'.format))

    # Figure 3a: Frequency of finished SMT tasks by solver timeout and feature-cardinality
    # threshold
    plot_results = print_results.copy()
    plot_results['param.timeout'] = plot_results['param.timeout'].astype(int)  # Int64 doesn't work
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.lineplot(x='param.timeout', y='finished', hue='param.k', style='param.k',
                 data=plot_results, palette=sns.color_palette(DEFAULT_COL_PALETTE, 7)[1:])
    plt.xlabel('Solver timeout in seconds')
    plt.xscale('log')
    plt.xticks(ticks=[2**x for x in range(12)],
               labels=['$2^{' + str(x) + '}$' if x % 2 == 1 else '' for x in range(12)])
    plt.xticks(ticks=[], minor=True)
    plt.ylabel('Finished tasks')
    plt.yticks(ticks=np.arange(start=0, stop=1.1, step=0.2))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    plt.ylim(-0.05, 1.05)
    leg = plt.legend(title='$k$', edgecolor='white', loc='upper left',
                     bbox_to_anchor=(-0.15, -0.1), columnspacing=1, framealpha=0, ncols=3)
    leg.get_title().set_position((-118, -33))
    plt.tight_layout()
    plt.savefig(plot_dir / 'csd-timeouts-finished-tasks.pdf')

    print('\n-- Subgroup quality --')

    eval_results = results[(results['sd_name'] == 'SMT') & (results['param.k'] == max_k) &
                           results['alt.number'].isin([pd.NA, 0]) &
                           results['param.tau_abs'].isin([pd.NA, min_tau_abs])]
    all_timeout_datasets = eval_results.groupby('dataset_name')['optimization_status'].agg(
        lambda x: (x != 'sat').all())  # returns Series with bool values and DS names as index
    all_timeout_datasets = all_timeout_datasets[all_timeout_datasets].index.to_list()

    print('\nWhat is the mean value of evaluation metrics for SMT with different solver timeouts',
          '(with maximum cardinality and all datasets)?')
    print(eval_results.groupby('param.timeout')[evaluation_metrics].mean().round(3))

    print('\nWhat is the mean value of evaluation metrics for SMT with different solver timeouts',
          '(with maximum cardinality and timeout-only datasets)?')
    print(eval_results[eval_results['dataset_name'].isin(all_timeout_datasets)].groupby(
        'param.timeout')[evaluation_metrics].mean().round(3))

    print('\nWhat is the mean value of evaluation metrics for beam search on each timeout-only',
          'dataset (with maximum cardinality)?')
    print_results = results[(results['sd_name'] == 'Beam') & (results['param.k'] == max_k) &
                            results['alt.number'].isin([pd.NA, 0]) &
                            results['param.tau_abs'].isin([pd.NA, min_tau_abs])]
    print(print_results[print_results['dataset_name'].isin(all_timeout_datasets)].groupby(
        'dataset_name')[evaluation_metrics].mean().reset_index(drop=True).round(3))

    print('\nWhat is the mean value of evaluation metrics for beam search overall',
          '(with maximum cardinality)?')
    print(print_results[evaluation_metrics].mean().round(3))

    # Figure 3b: Subgroup quality by solver timeout and train/test
    plot_results = eval_results[['param.timeout', 'train_nwracc', 'test_nwracc']]
    plot_results = plot_results.melt(id_vars=['param.timeout'], value_name='nWRAcc',
                                     value_vars=['train_nwracc', 'test_nwracc'], var_name='Split')
    plot_results['param.timeout'] = plot_results['param.timeout'].astype(int)  # Int64 doesn't work
    plot_results['Split'] = plot_results['Split'].str.replace('_nwracc', '')
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.lineplot(x='param.timeout', y='nWRAcc', hue='Split', style='Split', data=plot_results,
                 palette=DEFAULT_COL_PALETTE, seed=25)
    plt.xlabel('Solver timeout in seconds')
    plt.xscale('log')
    plt.xticks(ticks=[2**x for x in range(12)],
               labels=['$2^{' + str(x) + '}$' if x % 2 == 1 else '' for x in range(12)])
    plt.xticks(ticks=[], minor=True)
    plt.ylabel('Mean nWRAcc')
    plt.ylim(-0.05, 0.65)
    plt.yticks(np.arange(start=0, stop=0.7, step=0.1))
    leg = plt.legend(title='Split', edgecolor='white', loc='upper left',
                     bbox_to_anchor=(0, -0.1), columnspacing=1, framealpha=0, ncols=2)
    leg.get_title().set_position((-107, -21))
    plt.tight_layout()
    plt.savefig(plot_dir / 'csd-timeouts-nwracc.pdf')

    print('\n------ 6.3 Feature-Cardinality Constraints ------')
    # max timeout, no alternatives

    eval_results = results[results['param.timeout'].isin([pd.NA, max_timeout]) &
                           results['alt.number'].isin([pd.NA, 0]) &
                           results['param.tau_abs'].isin([pd.NA, min_tau_abs])]
    all_datasets = eval_results['dataset_name'].unique()
    no_timeout_datasets = eval_results[eval_results['sd_name'] == 'SMT'].groupby('dataset_name')[
        'optimization_status'].agg(lambda x: (x == 'sat').all())  # bool Series with names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()
    print('\nNumber of datasets without solver timeouts:', len(no_timeout_datasets))

    print('\nHow does the number of actually selected features differ from the prescribed "k"?')
    print(pd.crosstab(
        eval_results.loc[eval_results['param.k'] != max_k, 'param.k'],
        eval_results.loc[eval_results['param.k'] != max_k, 'selected_feature_idxs'].apply(
            len).rename('actually selected')))

    print('\nWhat is the mean value of evaluation metrics for different subgroup-discovery',
          'methods and feature-cardinality thresholds (for all datasets)?')
    for metric in evaluation_metrics:
        print(eval_results.groupby(['sd_name', 'param.k'])[metric].mean().reset_index().pivot(
            index='param.k', columns='sd_name').round(3))

    print('\nWhat is the mean value of evaluation metrics for different subgroup-discovery',
          'methods and feature-cardinality thresholds (for datasets without timeouts in SMT',
          'optimization)?')
    for metric in evaluation_metrics:
        print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
            ['sd_name', 'param.k'])[metric].mean().reset_index().pivot(
                index='param.k', columns='sd_name').round(3))

    print('\n-- Subgroup quality --')

    # Figures 4a-4d: Subgroup quality by subgroup-discovery method and feature-cardinality
    # threshold (subfigures: train/test and timeouts y/n)
    plot_results = eval_results[['dataset_name', 'sd_name', 'param.k',
                                 'train_nwracc', 'test_nwracc']].copy()
    plot_results['param.k'] = plot_results['param.k'].replace({max_k: 6})  # enable lineplot
    for metric, metric_name in [('train_nwracc', 'train nWRAcc'), ('test_nwracc', 'test nWRAcc')]:
        for (dataset_list, selection_name, y_max) in [
                (all_datasets, 'all-datasets', 0.65),
                (no_timeout_datasets, 'no-timeout-datasets', 0.75)]:
            plt.figure(figsize=(5, 6))
            plt.rcParams['font.size'] = 18
            sns.lineplot(data=plot_results[plot_results['dataset_name'].isin(dataset_list)],
                         x='param.k', y=metric, hue='sd_name', style='sd_name', palette='Dark2',
                         hue_order=sd_name_plot_order, style_order=sd_name_plot_order, seed=25,
                         errorbar=None)  # draw only mean, no confidence intervals (easier to read)
            plt.xlabel('Feature cardinality $k$')
            plt.xticks(ticks=range(1, 7), labels=(list(range(1, 6)) + [max_k]))
            plt.ylabel('Mean ' + metric_name)
            plt.ylim(-0.05, y_max)
            plt.yticks(np.arange(start=0, stop=(y_max + 0.05), step=0.1))
            plt.legend(title=None, edgecolor='white', loc='upper left',
                       bbox_to_anchor=(0, -0.25), columnspacing=1, framealpha=0, ncols=2)
            plt.figtext(x=0.14, y=0.20, s='Method', rotation='vertical')
            plt.tight_layout()
            plt.savefig(plot_dir /
                        f'csd-cardinality-{metric.replace("_", "-")}-{selection_name}.pdf')

    print('\n-- Runtime --')

    print('\n## Table 4: Mean runtime by subgroup-discovery method and feature-cardinality',
          'threshold ##\n')
    print_results = eval_results.groupby(['sd_name', 'param.k'])['fitting_time'].mean()
    print_results = print_results.reset_index().pivot(index='sd_name', columns='param.k',
                                                      values='fitting_time')
    print_results.index.name = None
    print_results.columns.name = '$k$'
    print(print_results.style.format('{:.1f}~s'.format).to_latex(hrules=True))

    print('\n------ 6.4 Alternative Subgroup Descriptions ------')
    # max timeout, fixed cardinality

    eval_results = results[results['alt.number'].notna()]
    no_timeout_datasets = eval_results[eval_results['sd_name'] == 'SMT'].groupby('dataset_name')[
        'optimization_status'].agg(lambda x: (x == 'sat').all())  # bool Series with names as index
    no_timeout_datasets = no_timeout_datasets[no_timeout_datasets].index.to_list()
    print('\nNumber of datasets without solver timeouts:', len(no_timeout_datasets))

    print('\nWhat is the mean value of evaluation metrics for different numbers of alternatives,',
          'dissimilarity thresholds, and subgroup-discovery methods (for all datasets)?')
    for metric in evaluation_metrics + alt_evaluation_metrics:
        print(eval_results.groupby(['sd_name', 'alt.number', 'param.tau_abs'])[metric].mean(
            ).reset_index().pivot(index=['sd_name', 'alt.number'], columns='param.tau_abs').round(3))

    print('\nWhat is the mean value of evaluation metrics for different numbers of alternatives,',
          'dissimilarity thresholds, and subgroup-discovery methods (for datasets without',
          'timeout in SMT optimization)?')
    for metric in evaluation_metrics + alt_evaluation_metrics:
        print(eval_results[eval_results['dataset_name'].isin(no_timeout_datasets)].groupby(
            ['sd_name', 'alt.number', 'param.tau_abs'])[metric].mean().reset_index().pivot(
                index=['sd_name', 'alt.number'], columns='param.tau_abs').round(3))

    print('\n-- Subgroup similarity --')

    # Figures 5a, 5b: Subgroup similarity by number of alternative, dissimilarity threshold,
    # and subgroup-discovery method (subfigures: similarity measure)
    plot_results = eval_results.copy()
    plot_results['alt.number'] = plot_results['alt.number'].astype(int)  # Int64 doesn't work
    plot_results['param.tau_abs'] = plot_results['param.tau_abs'].astype(int)
    plot_results.rename(columns={'sd_name': '_sd_name', 'param.tau_abs': '_param.tau_abs'},
                        inplace=True)  # underscore hides these labels in legend
    for metric, metric_name, yticks in [
            ('alt.hamming', 'Norm. Hamming sim.', np.arange(start=0.8, stop=1.05, step=0.1)),
            ('alt.jaccard', 'Jaccard sim.', np.arange(start=0.2, stop=1.05, step=0.1))]:
        plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = 18
        sns.lineplot(x='alt.number', y=metric, hue='_param.tau_abs', style='_sd_name',
                     data=plot_results, palette=sns.color_palette(DEFAULT_COL_PALETTE, 4)[1:],
                     seed=25)
        plt.xlabel('Number of alternative')
        plt.xticks(range(6))
        plt.ylabel(metric_name)
        plt.yticks(yticks)
        plt.legend(title=None, edgecolor='white', loc='upper left',
                   bbox_to_anchor=(0, -0.2), columnspacing=4, framealpha=0, ncols=2)
        plt.figtext(x=0.54, y=0.18, s='Method', rotation='vertical')
        plt.figtext(x=0.16, y=0.21, s='$\\tau_{\\mathrm{abs}}$', rotation='vertical')
        plt.tight_layout()
        plt.savefig(plot_dir / f'csd-alternatives-{metric.replace("alt.", "")}.pdf')

    print('\n-- Subgroup quality --')

    # Figures 6a, 6b: Subgroup quality by number of alternative, dissimilarity threshold, and
    # subgroup-discovery method (subfigures: train/test)
    for metric, metric_name in [('train_nwracc', 'train nWRAcc'), ('test_nwracc', 'test nWRAcc')]:
        plt.figure(figsize=(5, 5))
        plt.rcParams['font.size'] = 18
        sns.lineplot(x='alt.number', y=metric, hue='_param.tau_abs', style='_sd_name',
                     data=plot_results, palette=sns.color_palette(DEFAULT_COL_PALETTE, 4)[1:],
                     seed=25)
        plt.xlabel('Number of alternative')
        plt.xticks(range(6))
        plt.ylabel('Mean ' + metric_name)
        plt.ylim(-0.05, 0.65)
        plt.yticks(np.arange(start=0, stop=0.7, step=0.1))
        plt.legend(title=None, edgecolor='white', loc='upper left',
                   bbox_to_anchor=(0, -0.2), columnspacing=4, framealpha=0, ncols=2)
        plt.figtext(x=0.54, y=0.18, s='Method', rotation='vertical')
        plt.figtext(x=0.16, y=0.21, s='$\\tau_{\\mathrm{abs}}$', rotation='vertical')
        plt.tight_layout()
        plt.savefig(plot_dir / f'csd-alternatives-{metric.replace("_", "-")}.pdf')

    print('\nWhat is the mean value of evaluation metrics (shifted to [0, 1] and max-normalized',
          'with the quality of the original subgroup) for different numbers of alternatives,',
          'dissimilarity thresholds, and subgroup-discovery methods (for all datasets)?')
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

    print('\n-- Runtime --')

    print('\n## Table 5: Mean runtime by number of alternative, dissimilarity threshold, and',
          'subgroup-discovery method ##\n')
    print_results = eval_results.groupby(['sd_name', 'alt.number', 'param.tau_abs'])[
        'fitting_time'].mean()
    print_results = print_results.reset_index().pivot(index=['sd_name', 'param.tau_abs'],
                                                      columns='alt.number')
    print_results = print_results.droplevel(None, axis='columns')  # only included "fitting_time"
    print(print_results.style.format('{:.1f}~s'.format).to_latex(hrules=True, multirow_align='t'))

    print('\nWhat is the frequency of finished SMT tasks for different numbers of alternatives',
          'and dissimilarity thresholds?')
    print(eval_results[eval_results['sd_name'] == 'SMT'].groupby(['alt.number', 'param.tau_abs'])[
        'optimization_status'].agg(lambda x: (x == 'sat').sum() / len(x)).rename('').reset_index(
            ).pivot(index='alt.number', columns='param.tau_abs').applymap('{:.1%}'.format))


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates the paper\'s plots + tables and prints statistics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/datasets/', dest='data_dir',
                        help='Directory with prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    print('Evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('\nEvaluation finished. Plots created and saved.')
