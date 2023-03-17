from Preprocessing.MECO_data_split import split_into_rows, concat_MECO_langs
from basic_classifiers import grid_search_on_data, get_clf_cv_score, best_grid_search_params
from default_values import default_classifiers
import numpy as np
# import warnings
#
#
# def warn(*args, **kwargs):
#     pass
#
#
# warnings.warn = warn

data = concat_MECO_langs('ru')
datasets = {
    'ru_all_texts_fix': split_into_rows(data, cols='fix'),
    'ru_all_texts_demo': split_into_rows(data, cols='demo'),
    'ru_all_texts_fix+demo': split_into_rows(data, cols='fix+demo'),

    'ru_text2_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
    'ru_text2_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]}),

    'ru_text5_fix': split_into_rows(data, cols='fix', with_values_only={'Text_ID': [2]}),
    'ru_text5_fix+demo': split_into_rows(data, cols='fix+demo', with_values_only={'Text_ID': [2]})
}

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
        'solver': ['lbfgs'],
        'verbose': [1]
    },
    'abc': {
        'learning_rate': [0.1],
        'n_estimators': [100],
    },
    'rfc': {
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
        'n_jobs': [-1]
    },
    'gbc': {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    }
}


for data_name, (X, y) in data.items():
    gs_cv_results = grid_search_on_data(X, y, classifiers='all')

    print(f'Grid search results for {data_name}:\n')
    for clf, results in gs_cv_results.items():
        best_gs_params = best_grid_search_params(results)
        renamed_params = {}
        for param_name, param in best_gs_params.items():
            renamed_params[param_name] = param

        print(f'for parameters used on {clf}:')
        for param_name, param in renamed_params.items():
            print(f'\t{param_name} = {param}')
        print()

        cv_score = get_clf_cv_score(X, y, renamed_params)
        print('results below were obtained:')
        for score in cv_score:
            print(f'\t{score} (mean, std):',
                  f'({np.round(np.mean(cv_score[score]), 3)},',
                  f'{np.round(np.std(cv_score[score]), 3)})')
