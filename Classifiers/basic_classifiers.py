from default_values import default_scorers, default_params, default_classifiers

from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def best_grid_search_params(cv_results):
    search_params = cv_results['params'][0].keys()
    best_comb = np.argmax(cv_results['mean_test_score'])
    best_params = cv_results['params'][best_comb]
    best_dict = {}
    for param in search_params:
        best_dict[param[5:]] = best_params[f'{param}']
    return best_dict


def get_grid_search_params(X, y, classifier, params, n_jobs=4, n_splits=3, print_best=False):
    pipeline = GridSearchCV(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', classifier()),
        ]),
        param_grid=dict(zip([f'clf__{param}' for param in params], list(params.values()))),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), n_jobs=n_jobs
    )
    pipeline.fit(X, y)
    cv_results = pipeline.cv_results_

    if print_best:
        print('\nBest parameters by accuracy:')
        for param_name, param in best_grid_search_params(cv_results).items():
            print(f'{param_name}: {param}')

    return cv_results


def get_clf_cv_score(X, y, clf, params, cv_folds=10, n_jobs=4, scorers=None):
    if scorers is None:
        scorers = default_scorers
    cv_scores = cross_validate(clf(**params), X, y, cv=cv_folds, scoring=scorers, n_jobs=n_jobs)
    return cv_scores
