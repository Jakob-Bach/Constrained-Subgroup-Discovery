"""Run competitor-runtime evaluation

Script to compute summary statistics for the competitor-runtime experiments. Should be run after
the corresponding experimental pipeline, as this evaluation script requires the competitor-runtime
pipeline's outputs as inputs.

Usage: python -m run_competitor_runtime_evaluation --help
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

    print('\nWhat is the mean runtime for different subgroup-discovery methods and',
          'feature-cardinality thresholds?')
    print(results.groupby(['sd_name', 'param.k'], as_index=False)['fitting_time'].mean().pivot(
        index='sd_name', columns='param.k').round(2))


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
