import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Preprocessing.MECO_data_split import split_into_rows, concat_MECO_langs
from basic_classifiers import grid_search_on_data, get_clf_cv_score, best_grid_search_params
from default_values import default_classifiers

import numpy as np


def make_datasets(*langs, path_to_data):
    data = concat_MECO_langs(*langs, path_to_data=path_to_data)
    name_prefix = '_'.join(langs) + '_'
    datasets = {
        name_prefix + 'all_texts_fix': split_into_rows(data, cols='fix'),
        # name_prefix + 'all_texts_demo': split_into_rows(data, cols='demo'),
        # name_prefix + 'all_texts_fix+demo': split_into_rows(data, cols='fix+demo'),
        #
        # name_prefix + 'text2_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
        # name_prefix + 'text2_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]}),
        #
        # name_prefix + 'text5_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
        # name_prefix + 'text5_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]})
    }
    return datasets


def run_check_grid_search_for_datasets(datasets, classifiers=None, params=None, n_jobs=4, output=sys.stdout):
    for data_name, (X, y) in datasets.items():
        print(f'Grid search results for {data_name}:', file=output)
        gs_cv_results = grid_search_on_data(X, y, classifiers=classifiers, params=params, n_jobs=n_jobs)
        for clf_name, results in gs_cv_results.items():
            cv_params = best_grid_search_params(results)

            print(f'for parameters used on {clf_name}:', file=output)
            for param_name, param in cv_params.items():
                print(f'\t{param_name}', f'= {param}', file=output)

            cv_score = get_clf_cv_score(X, y, default_classifiers[clf_name], cv_params, n_jobs=n_jobs)
            print('results below were obtained (mean, std):', file=output)
            for score in cv_score:
                print(f'\t{score}:',
                      f'({np.round(np.mean(cv_score[score]), 3)},',
                      f'{np.round(np.std(cv_score[score]), 3)})', file=output)
        print(file=output)
