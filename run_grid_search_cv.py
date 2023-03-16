import warnings
from basic_classifiers import *


def warn(*args, **kwargs):
    pass


warnings.warn = warn


data = {
    'ru_all_texts_fix': getXy('ru', cols='fix'),
    'ru_all_texts_demo': getXy('ru', cols='demo'),

    'ru_text2_fix': getXy('ru', cols='fix', text_id=2),
    'ru_text2_demo': getXy('ru', cols='demo', text_id=2),

    'ru_text5_fix': getXy('ru', cols='fix', text_id=5),
    'ru_text5_demo': getXy('ru', cols='demo', text_id=5)
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
    'ab': {
        'learning_rate': [0.1],
        'n_estimators': [100],
    },
    'rf': {
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
        'n_jobs': [-1]
    },
    'gb': {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    }
}


for data_name, (X, y) in data.items():
    gs_cv_results = grid_search_on_data(X, y, classifiers='all', params=default_clf_params)

    print(f'Grid search results for {data_name}:\n')
    for clf, results in gs_cv_results.items():
        best_gs_params = get_best_params(results)
        renamed_params = {}
        for param_name, param in best_gs_params.items():
            renamed_params[param_name] = param

        print(f'for parameters used on {clf}:')
        for param_name, param in renamed_params.items():
            print(f'\t{param_name} = {param}')
        print()

        cv_score = get_cv_score(X, y, clf, renamed_params)
        print('results below were obtained:')
        for score in cv_score:
            print(f'\t{score} (mean, std):',
                  f'({np.round(np.mean(cv_score[score]), 3)},',
                  f'{np.round(np.std(cv_score[score]), 3)})')
