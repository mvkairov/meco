from default_values import default_scorers, default_clf_params, default_classifiers

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


def grid_search_on_classifier(classifier, X, y, search_params, n_splits=3, n_jobs=4, print_best=False):
    pipeline = GridSearchCV(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', classifier),
        ]),
        param_grid=dict(zip([f'clf__{param}' for param in search_params], list(search_params.values()))),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), n_jobs=n_jobs
    )

    pipeline.fit(X, y)
    cv_results = pipeline.cv_results_

    if print_best:
        print('\nBest parameters by accuracy:')
        for param_name, param in best_grid_search_params(cv_results).items():
            print(f'{param_name}: {param}')

    return cv_results


def grid_search_on_data(X, y, classifiers=None, params=None, n_jobs=4):
    if classifiers == 'all' or classifiers is None:
        classifiers = default_classifiers

    if params is None:
        params = default_clf_params
    for clf in classifiers:
        if clf not in params.keys():
            params[clf] = default_clf_params[clf]

    results = {}
    for clf in classifiers:
        results[clf] = grid_search_on_classifier(classifiers[clf](), X, y, params[clf], n_jobs=n_jobs)
    return results


def get_clf_cv_score(X, y, clf, params, cv_folds=10, n_jobs=4, scorers=None):
    if scorers is None:
        scorers = default_scorers
    cv_scores = cross_validate(clf(**params), X, y, cv=cv_folds, scoring=scorers, n_jobs=n_jobs)
    return cv_scores
