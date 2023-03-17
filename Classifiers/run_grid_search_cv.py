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
from termcolor import colored
import colorama
colorama.init()


def make_datasets(*langs):
    data = concat_MECO_langs(*langs)
    name_prefix = '_'.join(langs) + '_'
    datasets = {
        # 'all_texts_fix': split_into_rows(data, cols='fix'),
        # name_prefix + 'all_texts_demo': split_into_rows(data, cols='demo'),
        name_prefix + 'all_texts_fix+demo': split_into_rows(data, cols='fix+demo'),
        #
        # name_prefix + 'text2_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
        # name_prefix + 'text2_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]}),
        #
        # name_prefix + 'text5_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
        # name_prefix + 'text5_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]})
    }
    return datasets


def run_check_grid_search_for_datasets(datasets, classifiers, params=None):
    for data_name, (X, y) in datasets.items():
        print(colored(f'Grid search results for {data_name}:', 'light_yellow', 'on_magenta'))
        gs_cv_results = grid_search_on_data(X, y, classifiers=classifiers, params=params)
        for clf_name, results in gs_cv_results.items():
            cv_params = best_grid_search_params(results)

            print(colored(f'for parameters used on {clf_name}:', 'cyan'))
            for param_name, param in cv_params.items():
                print(colored(f'\t{param_name}', 'green'), f'= {param}')

            cv_score = get_clf_cv_score(X, y, default_classifiers[clf_name], cv_params)
            print(colored('results below were obtained (mean, std):', 'cyan'))
            for score in cv_score:
                print(colored(f'\t{score}:', 'green'),
                      f'({np.round(np.mean(cv_score[score]), 3)},',
                      f'{np.round(np.std(cv_score[score]), 3)})')
        print()


# to be used for debugging
test_params = {
    'knn': {
        'n_neighbors': [10],
        'p': [5],
    },
    'mlp': {
        'hidden_layer_sizes': [100],
        'max_iter': [10000],
        'learning_rate_init': [10 ** (-3)],
        'activation': ['relu'],
        'solver': ['adam'],
    },
    'abc': {
        'learning_rate': [0.1],
        'n_estimators': [100],
    },
    'rfc': {
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    },
    'gbc': {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    }
}

ru_datasets = make_datasets('ru')
run_check_grid_search_for_datasets(ru_datasets)
