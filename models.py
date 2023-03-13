from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import numpy as np

fix_cols = ['Fix_X', 'Fix_Y', 'Fix_Duration']
demo_cols = ['motiv', 'IQ', 'Age', 'Sex']


def print_grid_search_results(search_params, n_splits, cv_results):
    header_str = '|'
    columns = list(search_params.keys())
    columns += [f'score_fold_{i + 1}' for i in range(n_splits)]
    for col in columns:
        header_str += '{:^12}|'.format(col)
    print(header_str)
    print('-' * (len(columns) * 13))

    for i in range(len(cv_results['params'])):
        s = '|'
        for param in search_params.keys():
            s += '{:>12}|'.format(cv_results['params'][i][f'clf__{param}'])
        for k in range(n_splits):
            score = cv_results[f'split{k}_test_score'][i]
            score = np.around(score, 5)
            s += '{:>12}|'.format(score)
        print(s.strip())


def clf_grid_search(classifier, X, y, search_params, n_splits=3, print_results=False):
    pipeline = GridSearchCV(
        Pipeline([
            # ('minmax', MinMaxScaler()),
            ('clf', classifier),
        ]),
        dict(zip([f'clf__{param}' for param in search_params], list(search_params.values()))),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    )

    pipeline.fit(X, y)
    results = pipeline.cv_results_

    best_comb = np.argmax(results['mean_test_score'])
    best_params = results['params'][best_comb]
    best_dict = {}
    for param in search_params:
        best_dict[param] = best_params[f'clf__{param}']

    if print_results:
        print_grid_search_results(search_params, n_splits, results)

    return best_dict
