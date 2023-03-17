from Preprocessing.MECO_data_split import MECODataSplit, concat_MECO_langs
from sklearn.utils import shuffle
from keras.utils import to_categorical

# from keras import Sequential
# from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
# from scikeras.wrappers import KerasClassifier

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

df = concat_MECO_langs('ru', path_to_data='../Datasets/DataToUse/')
dataset = MECODataSplit(df)
X, y = dataset.split_by_unique_values(split_cols=['SubjectID'],
                                      include_cols='fix',
                                      test_size=0, fix_threshold=0,
                                      resample='truncate', series_length=800)

y = to_categorical(y)
X, y = shuffle(X, y)
n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], X.shape[1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]


# def make_CNN_model(n_timesteps, n_features, n_outputs, conv_filters=64,
#                    kernel_size=3, strides=1, dropout_rate=0.5, pool_size=2,
#                    loss='categorical_crossentropy', optimizer='adam', metrics=None):
#     if metrics is None:
#         metrics = ['accuracy']
#
#     model = Sequential()
#     model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=strides,
#                      activation='relu', input_shape=(n_timesteps, n_features)))
#     model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, strides=strides, activation='relu'))
#     model.add(MaxPooling1D(pool_size=pool_size))
#     model.add(Dropout(dropout_rate))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#
#     model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#     return model
#
#
# def CNN_model_wrapper(n_timesteps=None, n_features=None, n_outputs=None,
#                       conv_filters=64, kernel_size=3, strides=1, dropout_rate=0.5, pool_size=2,
#                       loss='categorical_crossentropy', optimizer='adam', metrics=None):
#     if metrics is None:
#         metrics = ['accuracy']
#
#     sk_wrapped_model = KerasClassifier(make_CNN_model,
#                                        n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs,
#                                        conv_filters=conv_filters, kernel_size=kernel_size, strides=strides,
#                                        dropout_rate=dropout_rate, pool_size=pool_size,
#                                        loss=loss, optimizer=optimizer, metrics=metrics)
#     return sk_wrapped_model
