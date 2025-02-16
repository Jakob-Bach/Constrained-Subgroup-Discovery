"""Run SIGMOD competitor-runtime evaluation

Script to compute summary statistics and create tables for the competitor-runtime experiments for
the SIGMOD revision phase (used in the cover letter, but did not make it into the actual paper).
Should be run after the corresponding experimental pipeline, as this evaluation script requires the
competitor-runtime pipeline's outputs as inputs.

Usage: python -m run_competitor_runtime_evaluation_sigmod --help
"""


import argparse
import pathlib

import data_handling


# Main routine: Run competitor-runtime evaluation pipeline. To this end, read results from the
# "results_dir" and print some statistics to the console.
def evaluate(results_dir: pathlib.Path) -> None:
    if not results_dir.is_dir():
        raise FileNotFoundError('The results directory does not exist.')

    results = data_handling.load_results(directory=results_dir)

    print('\nHow many results are there for each subgroup-discovery method on each dataset?',
          f'(with {results["split_idx"].nunique()} folds and {results["param.k"].nunique()}',
          'feature-cardinality thresholds)')
    print_results = results.groupby(['dataset_name', 'sd_name'], as_index=False).size()
    print_results = print_results.pivot(index='sd_name', columns='dataset_name', values='size')
    print_results = print_results.fillna(0).astype(int)
    print(print_results)

    print('\n## Table 1 in SIGMOD cover letter: Which subgroup-discovery method finished on which',
          'dataset? ##\n')  # in given data, either on all folds or none
    print_results = (print_results == print_results.max())  # compare each value to column maximum
    print_results = print_results.rename(columns=(lambda x: x.replace('_', '\\_')))  # for LaTeX
    print_results = print_results.replace({False: '', True: '\\checkmark'})  # for LaTeX
    print_results.index.name = None  # row only containing "sd_name"
    print_results.columns.name = 'Method'
    print(print_results.style.to_latex(hrules=True))

    print('\nWhat is the mean runtime for different subgroup-discovery methods and',
          'feature-cardinality thresholds on each dataset?')
    for dataset_name in results['dataset_name'].unique():
        print('\nDataset:', dataset_name)
        print_results = results[results['dataset_name'] == dataset_name].groupby(
            ['sd_name', 'param.k'], as_index=False)['fitting_time'].mean()
        print_results = print_results.pivot(index='sd_name', columns='param.k',
                                            values='fitting_time')
        print(print_results.round(2))

    print('\n## Table 2 in SIGMOD cover letter: Mean runtime by subgroup-discovery method and',
          'feature-cardinality threshold on the dataset "spect" ##\n')
    print_results = print_results.rename(columns=(lambda x: f'$k={x}$'))
    print_results.index.name = None  # row only containing "sd_name"
    print_results.columns.name = 'Method'
    print(print_results.style.format(precision=2).to_latex(hrules=True))


# Parse some command-line arguments and run the main routine.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prints statistics for the competitor-runtime experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--results', type=pathlib.Path, dest='results_dir',
                        default='data/competitor-runtime-results/',
                        help='Directory with experimental results.')
    print('Competitor-runtime evaluation started.')
    evaluate(**vars(parser.parse_args()))
    print('\nCompetitor-runtime evaluation finished.')
