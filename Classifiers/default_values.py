from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

knn_n_neighbors = list(range(1, 11))
knn_p_norm = [x / 20 for x in range(20, 101)]  # [1, 5] range with 0.05-sized steps

mlp_neurons = list(range(10, 201, 10)) + list((i, i) for i in range(10, 201, 10))
mlp_max_iter = list(range(1000, 10000, 1000)) + list(range(10000, 50001, 5000))
mlp_lr = [10 ** i for i in range(-6, -1)]
mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
mlp_solver = ['sgd', 'adam']

rfc_estimators = list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))
rfc_min_samples_split = list(range(2, 11))
rfc_min_samples_leaf = list(range(1, 11))
gbc_lr = [10 ** (-3), 10 ** (-2)] + [i / 10 for i in range(1, 6)]


default_clf_params = {
    'knn': {
        'n_neighbors': knn_n_neighbors,
        'p': knn_p_norm
    },
    'mlp': {
        'hidden_layer_sizes': mlp_neurons,
        'max_iter': mlp_max_iter,
        'learning_rate_init': mlp_lr,
        'activation': mlp_activation,
        'solver': mlp_solver,
    },
    'abc': {
        'learning_rate': gbc_lr,
        'n_estimators': rfc_estimators,
    },
    'rfc': {
        'n_estimators': rfc_estimators,
        'min_samples_split': rfc_min_samples_split,
        'min_samples_leaf': rfc_min_samples_leaf,
    },
    'gbc': {
        'learning_rate': gbc_lr,
        'n_estimators': rfc_estimators,
        'min_samples_split': rfc_min_samples_split,
        'min_samples_leaf': rfc_min_samples_leaf,
    }
}

default_classifiers = {
    'knn': KNeighborsClassifier,
    'mlp': MLPClassifier,
    # 'abc': AdaBoostClassifier,
    'rfc': RandomForestClassifier,
    # 'gbc': GradientBoostingClassifier
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
