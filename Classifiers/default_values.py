from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

mlp_neurons = list(range(10, 201, 10)) + list((i, i) for i in range(10, 201, 10))
mlp_max_iter = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + \
             list(range(1000, 10000, 1000)) + list(range(10000, 50001, 5000))
mlp_lr = [10 ** i for i in range(-6, -1)]

rfc_estimators = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))
gbc_lr = [10 ** (-3), 10 ** (-2)] + [i / 10 for i in range(1, 6)]

default_clf_params = {
    'knn': {
        'n_neighbors': list(range(1, 11)),
        'p': list(range(1, 6)),
    },
    'mlp': {
        'hidden_layer_sizes': mlp_neurons,
        'max_iter': mlp_max_iter,
        'learning_rate_init': mlp_lr,
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
    },
    'abc': {
        'learning_rate': gbc_lr,
        'n_estimators': rfc_estimators,
    },
    'rfc': {
        'n_estimators': rfc_estimators,
        'min_samples_split': list(range(2, 11)),
        'min_samples_leaf': list(range(1, 11)),
    },
    'gbc': {
        'learning_rate': gbc_lr,
        'n_estimators': rfc_estimators,
        'min_samples_split': list(range(2, 11)),
        'min_samples_leaf': list(range(1, 11)),
    }
}

default_classifiers = {
    'knn': KNeighborsClassifier,
    'mlp': MLPClassifier,
    'abc': AdaBoostClassifier,
    'rfc': RandomForestClassifier,
    'gbc': GradientBoostingClassifier
}


# StratifiedKFold doesn't support one-hot encoding, while roc_auc requires it
def custom_roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr'):
    y_true = label_binarize(y_true, classes=list(range(6)))
    y_pred = label_binarize(y_pred, classes=list(range(6)))
    return roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class)


default_scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='micro'),
    'precision_score': make_scorer(precision_score, average='micro'),
    'recall_score': make_scorer(recall_score, average='micro'),
    'roc_auc_score': make_scorer(custom_roc_auc_score, average='micro', multi_class='ovr')
}
