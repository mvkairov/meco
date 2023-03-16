from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from keras import Sequential
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from scikeras.wrappers import KerasClassifier

import numpy as np

fix_cols = ['Fix_X', 'Fix_Y', 'Fix_Duration']
demo_cols = ['motiv', 'IQ', 'Age', 'Sex']


def print_grid_search_results(n_splits, cv_results):
    header_str = '|'
    search_params = cv_results['params'].keys()
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


def get_best_params(cv_results):
    search_params = cv_results['params'][0].keys()
    best_comb = np.argmax(cv_results['mean_test_score'])
    best_params = cv_results['params'][best_comb]
    best_dict = {}
    for param in search_params:
        best_dict[param[5:]] = best_params[f'{param}']
    return best_dict


def clf_grid_search(classifier, X, y, search_params, n_splits=3, print_best=False, print_results=False):
    pipeline = GridSearchCV(
        estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('clf', classifier),
        ]),
        param_grid=dict(zip([f'clf__{param}' for param in search_params], list(search_params.values()))),
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), n_jobs=-1
    )

    pipeline.fit(X, y)
    cv_results = pipeline.cv_results_

    if print_results:
        print_grid_search_results(n_splits, cv_results)
    if print_best:
        print('\nBest parameters by accuracy:')
        for param_name, param in get_best_params(cv_results).items():
            print(f'{param_name}: {param}')

    return cv_results


def make_CNN_model(n_timesteps, n_features, n_outputs, conv_filters=64,
                   kernel_size=3, strides=1, dropout_rate=0.5, pool_size=2,
                   loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):

    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=strides,
                     activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=strides, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def CNN_model_wrapper(n_timesteps=None, n_features=None, n_outputs=None,
                      conv_filters=64, kernel_size=3, strides=1, dropout_rate=0.5, pool_size=2,
                      loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):

    sk_wrapped_model = KerasClassifier(make_CNN_model,
                                       n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs,
                                       conv_filters=conv_filters, kernel_size=kernel_size, strides=strides,
                                       dropout_rate=dropout_rate, pool_size=pool_size,
                                       loss=loss, optimizer=optimizer, metrics=metrics)
    return sk_wrapped_model

