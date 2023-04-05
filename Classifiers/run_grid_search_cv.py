import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Preprocessing.MECO_data_split import split_into_rows, concat_MECO_langs
from basic_classifiers import get_grid_search_params, get_clf_cv_score, best_grid_search_params


def make_dataset(langs, cols='fix', params=None, path_to_data='Datasets/DataToUse/'):
    data = concat_MECO_langs(langs, path_to_data=path_to_data)
    X, y = split_into_rows(data, cols=cols, with_values_only=params)
    return X, y


def run_grid_search_cv(X, y, classifier, params, n_jobs=4):
    gs_cv_results = get_grid_search_params(X, y, classifier, params, n_jobs=n_jobs)
    best_gs_params = best_grid_search_params(gs_cv_results)
    cv_score = get_clf_cv_score(X, y, classifier, best_gs_params, n_jobs=n_jobs)
    return best_gs_params, cv_score
