from models import *
from MECO_data_split import *
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

rf_estimators = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))
gb_lr = [10 ** (-3), 10 ** (-2)] + [i / 10 for i in range(1, 6)]

default_clf_params = {
    'knn': {
        'n_neighbors': list(range(1, 11)),
        'p': list(range(1, 6)),
    },
    'mlp': {
        'hidden_layer_sizes': list(range(10, 201, 10)),
        'max_iter': list(range(10000, 50001, 5000)),
        'learning_rate_init': [10 ** i for i in range(-6, -1)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
    },
    'ab': {
        'learning_rate': gb_lr,
        'n_estimators': rf_estimators,
    },
    'rf': {
        'n_estimators': rf_estimators,
        'min_samples_split': list(range(2, 11)),
        'min_samples_leaf': list(range(1, 11)),
        'n_jobs': [-1]
    },
    'gb': {
        'learning_rate': gb_lr,
        'n_estimators': rf_estimators,
        'min_samples_split': list(range(2, 11)),
        'min_samples_leaf': list(range(1, 11)),
    }
}

clf_dict = {
    'knn': KNeighborsClassifier,
    'mlp': MLPClassifier,
    'ab': AdaBoostClassifier,
    'rf': RandomForestClassifier,
    'gb': GradientBoostingClassifier
}

default_scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='micro'),
    'precision_score': make_scorer(precision_score, average='micro'),
    'recall_score': make_scorer(recall_score, average='micro'),
    'roc_auc_score': make_scorer(roc_auc_score, average='micro', multi_class='ovr')
}


def getXy(*langs, cols='demo', text_id=None):
    data = concat_MECO_langs(*langs, path_to_data='Datasets/DataToUse/')
    if cols == 'fix':
        cols = fix_cols
    elif cols == 'demo':
        cols = fix_cols + demo_cols

    if text_id is not None:
        X = data[data['Text_ID'] == text_id][cols]
        y = data[data['Text_ID'] == text_id]['Target_Label']
    else:
        X = data[cols]
        y = data['Target_Label']

    y = label_binarize(y, classes=list(range(6)))
    X, y = shuffle(X, y)
    return X, y


def grid_search_on_data(X, y, classifiers=None, params=None):
    if classifiers == 'all' or classifiers is None:
        classifiers = list(clf_dict.keys())
    # elif any([clf in clf_dict.keys() for clf in classifiers]):
    #     print('Error: unknown classifier listed! Use \'all\' to run on all available classifiers, or pass a list.')
    #     raise

    if params is None:
        params = default_clf_params
    for clf in classifiers:
        if clf not in params.keys():
            params[clf] = default_clf_params[clf]

    results = {}
    for clf in classifiers:
        results[clf] = clf_grid_search(clf_dict[clf](), X, y, params[clf])
    return results


def get_cv_score(X, y, clf_name, params, cv_folds=10, scorers=None):
    classifier = clf_dict[clf_name](**params)

    if scorers is None:
        scorers = default_scorers
    cv_scores = cross_validate(classifier, X, y, cv=cv_folds, scoring=scorers)
    # for score in cv_scores:
    #     print(f'{score} (mean, std): ({np.round(np.mean(cv_scores[score]), 3)}, \
    #                                   {np.round(np.std(cv_scores[score]), 3)})')
    return cv_scores
