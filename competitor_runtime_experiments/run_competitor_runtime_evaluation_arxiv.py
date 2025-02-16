"""Run arXiv competitor-runtime evaluation

Script to compute summary statistics and create tables for the competitor-runtime experiments for
the arXiv version of the paper. Should be run after the corresponding experimental pipeline, as
this evaluation script requires the competitor-runtime pipeline's outputs as inputs.

Usage: python -m run_competitor_runtime_evaluation_arxiv --help
"""


import argparse
import pathlib

import data_handling


# Main routine: Run competitor-runtime evaluation pipeline. To this end, read results from the
# "results_dir" and some dataset information from "data_dir". Print some statistics and LaTeX-ready
# tables to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('The results directory does not exist.')

    results = data_handling.load_results(directory=results_dir)
    dataset_overview = data_handling.load_dataset_overview(directory=data_dir)

    print('\n-------- A Appendix --------')

    print('\n------ A.3 Competitor-Runtime Experiments ------')

    print('\n---- A.3.1 Experimental Design ----')

    print('\n-- Datasets --')

    print('\n## Table 6: Dataset overview ##\n')
    print_results = dataset_overview.rename(columns={'dataset': 'Dataset', 'n_instances': '$m$',
                                                     'n_features': '$n$'})
    print_results.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(print_results.style.format(escape='latex').hide(axis='index').to_latex(hrules=True))

    print('\n---- A.3.2 Evaluation ----')

    print('\n-- Timeouts --')

    print('\nHow many results are there for each subgroup-discovery method on each dataset?',
          f'(with {results["split_idx"].nunique()} folds and {results["param.k"].nunique()}',
          'feature-cardinality thresholds)')
    print_results = results.groupby(['dataset_name', 'sd_name'], as_index=False).size()
    print_results = print_results.pivot(index='sd_name', columns='dataset_name', values='size')
    print_results = print_results.fillna(0).astype(int)
    print(print_results)

    print('\n## Table 7: Which subgroup-discovery method finished on which dataset? ##\n')
    print_results = (print_results == print_results.max())  # compare each value to column maximum
    print_results = print_results.rename(columns={
        'backache': 'back.', 'horse_colic': 'horse.', 'ionosphere': 'iono.'})
    print_results = print_results.replace({False: '', True: '\\checkmark'})  # for LaTeX
    print_results.index.name = None  # row only containing "sd_name"
    print_results.columns.name = 'Method'
    print(print_results.style.to_latex(hrules=True))

    print('\n-- Runtime --')

    print('\nWhat is the mean runtime for different subgroup-discovery methods and',
          'feature-cardinality thresholds on each dataset?')
    for dataset_name in results['dataset_name'].unique():
        print('\nDataset:', dataset_name)
        print_results = results[results['dataset_name'] == dataset_name].groupby(
            ['sd_name', 'param.k'], as_index=False)['fitting_time'].mean()
        print_results = print_results.pivot(index='sd_name', columns='param.k',
                                            values='fitting_time')
        print(print_results.round(2))

    print('\n## Table 8: Mean runtime by subgroup-discovery method and feature-cardinality',
          'threshold on the dataset "spect" ##\n')
    print_results = print_results.rename(columns=(lambda x: f'$k={x}$'))
    print_results.index.name = None  # row only containing "sd_name"
    print_results.columns.name = 'Method'
    print(print_results.style.format(precision=2).to_latex(hrules=True))


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prints statistics and tables for the competitor-runtime experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, dest='data_dir',
                        default='data/competitor-runtime-datasets/',
                        help='Directory with prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, dest='results_dir',
                        default='data/competitor-runtime-results/',
                        help='Directory with experimental results.')
    print('Competitor-runtime evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('\nCompetitor-runtime evaluation finished.')
