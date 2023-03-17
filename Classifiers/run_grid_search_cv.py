from Preprocessing.MECO_data_split import split_into_rows, concat_MECO_langs
from basic_classifiers import grid_search_on_data, get_clf_cv_score, best_grid_search_params
from default_values import default_classifiers
import numpy as np


def make_datasets(*langs):
    data = concat_MECO_langs(*langs)
    datasets = {
        'all_texts_fix': split_into_rows(data, cols='fix'),
        'all_texts_demo': split_into_rows(data, cols='demo'),
        'all_texts_fix+demo': split_into_rows(data, cols='fix+demo'),

        'text2_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
        'text2_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]}),

        'text5_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
        'text5_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]})
    }
    return datasets


def run_check_grid_search_for_datasets(datasets, classifiers='all', params=None):
    for data_name, (X, y) in datasets.items():
        gs_cv_results = grid_search_on_data(X, y, classifiers=classifiers, params=params)

        print(f'Grid search results for {data_name}:\n')
        for clf_name, results in gs_cv_results.items():
            cv_params = best_grid_search_params(results)

            print(f'for parameters used on {clf_name}:')
            for param_name, param in cv_params.items():
                print(f'\t{param_name} = {param}')

            cv_score = get_clf_cv_score(X, y, default_classifiers[clf_name], cv_params)
            print('results below were obtained:')
            for score in cv_score:
                print(f'\t{score} (mean, std):',
                      f'({np.round(np.mean(cv_score[score]), 3)},',
                      f'{np.round(np.std(cv_score[score]), 3)})')


ru_datasets = make_datasets('ru')
run_check_grid_search_for_datasets(ru_datasets)
