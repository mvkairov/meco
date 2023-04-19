from Models.default_values import *

from sklearn.model_selection import cross_validate, GridSearchCV, KFold, StratifiedKFold, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def best_gs_params(gs_results):
    search_params = gs_results['params'][0].keys()
    best_comb = np.argmax(gs_results['mean_test_score'])
    best_params = gs_results['params'][best_comb]
    best_dict = {}
    for param in search_params:
        best_dict[param.removeprefix('model__')] = best_params[f'{param}']
    return best_dict


def get_custom_cv(n_splits, value_indices=None):
    if value_indices is None:
        return StratifiedKFold(n_splits=n_splits)
    else:
        return PredefinedSplit(value_indices // n_splits)


def get_gs_params(X, y, model, params, value_indices=None, n_jobs=4, n_splits=3):
    pipeline = GridSearchCV(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('model', model()),
        ]),
        param_grid=dict(zip([f'model__{param}' for param in params], list(params.values()))),
        cv=StratifiedKFold(n_splits), n_jobs=n_jobs
    )
    pipeline.fit(X, y)
    gs_results = pipeline.cv_results_
    best_params = best_gs_params(gs_results)
    return best_params


def get_cv_score(X, y, model, params, scorers, value_indices=None, n_jobs=4, cv_folds=10):
    cv_scores = cross_validate(model(**params), X, y, cv=cv_folds,
                               scoring=scorers, n_jobs=n_jobs)
    for name, value in cv_scores.items():
        cv_scores[name] = (np.round(np.mean(value), 3), np.round(np.std(value), 3))
    return cv_scores


def gs_cv_results(X, y, model_name, method='Classification', value_indices=None, params=None, n_jobs=4, debug=False):
    if params is None:
        if debug:
            params = test_gs_params[model_name]
        else:
            params = default_gs_params[model_name]
    if method == 'Classification':
        model = default_classifiers[model_name]
        scorers = classification_scorers
    else:
        model = default_regressors[model_name]
        scorers = regression_scorers

    gs_results = get_gs_params(X, y, model, params, value_indices, n_jobs=n_jobs)
    cv_results = get_cv_score(X, y, model, gs_results, n_jobs=n_jobs, scorers=scorers, value_indices=None)
    return gs_results, cv_results
