from models import *
from MECO_data_split import *
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np


def getXy(*langs, text_id=None, cols):
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

    # y = label_binarize(y, classes=list(range(6)))
    X, y = shuffle(X, y)
    return X, y


X, y = getXy('ru', text_id=None, cols='demo')

# knn_res = clf_grid_search(KNeighborsClassifier(), X, y,
#                           {
#                               'n_neighbors': list(range(1, 11)),
#                               'p': list(range(1, 6)),
#                           })
# print(knn_res)

# mlp_res = clf_grid_search(MLPClassifier(), X, y,
#                           {
#                               'hidden_layer_sizes': [20],  # list(range(10, 201, 10)),
#                               'max_iter': [20000],  # list(range(10000, 50001, 5000)),
#                               'learning_rate_init': [10 ** i for i in range(-6, -1)],
#                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
#                               'solver': ['sgd', 'adam', 'lbfgs'],
#                               'verbose': [1],
#
#                           })
# print(mlp_res)

# rf_res = clf_grid_search(RandomForestClassifier(), X, y,
#                          {
#                              'n_estimators': [1000],  # list(range(10, 10011, 500)),
#                              'min_samples_split': [5],  # list(range(2, 11)),
#                              'min_samples_leaf': [5],  # list(range(1, 11)),
#                              'verbose': [1],
#                              'n_jobs': [4]
#                          })
# print(rf_res)

# ru_demo dataset

scorers = {
        'accuracy_score': make_scorer(accuracy_score),
        # 'f1_score': make_scorer(f1_score, average='micro'),
        # 'precision_score': make_scorer(precision_score, average='micro'),
        # 'recall_score': make_scorer(recall_score, average='micro'),
        # 'roc_auc_score': make_scorer(roc_auc_score, average='micro')
      }

clf = KNeighborsClassifier(n_neighbors=9, p=1)
cv_score = cross_validate(clf, X, y, cv=10, scoring=scorers)
for score in cv_score:
    print(f'{score} (mean, std): ({round(np.mean(cv_score[score]), 3)}, {round(np.std(cv_score[score]), 3)})')

