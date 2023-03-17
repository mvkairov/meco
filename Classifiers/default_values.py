from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

mlp_neurons = list(range(10, 201, 10)) + list((i, i) for i in range(10, 201, 10))
mlp_epochs = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + \
             list(range(1000, 10000, 1000)) + list(range(10000, 50001, 10000))
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
        'max_iter': mlp_epochs,
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
        'n_jobs': [-1]
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

default_scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='micro'),
    'precision_score': make_scorer(precision_score, average='micro'),
    'recall_score': make_scorer(recall_score, average='micro'),
    'roc_auc_score': make_scorer(roc_auc_score, average='micro', multi_class='ovr')
}