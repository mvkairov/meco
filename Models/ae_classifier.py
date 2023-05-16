import sys
sys.path.append('../')

import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, Conv1D, MaxPooling1D, Conv1DTranspose
from Preprocessing.MECO_data_split import concat_MECO_langs, split_into_time_series, lang_list
import numpy as np
from sklearn.metrics import r2_score
# tf.config.set_visible_devices([], 'GPU')
from Models.search_methods import search_cv
from run_cmd import make_msg

np.int = int


data = concat_MECO_langs(lang_list)
X, X_test, y, y_test, demo, demo_test = split_into_time_series(data, length=180)
X = (X - np.mean(X)) / np.std(X)

X_dur = X.T[2].T
X_test_dur = X_test.T[2].T


input_img = Input(shape=(180,))
encoder = Dense(128, activation='relu')(input_img)
encoder = Dense(64, activation='relu')(encoder)

encoder = Dense(32, activation='relu')(encoder)

decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(128, activation='relu')(decoder)
output_img = Dense(180, activation='relu')(decoder)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_dur, np.float32(X_dur), epochs=10)

X_pred = autoencoder.predict(X_test_dur)
print(r2_score(X_test_dur, X_pred))

enc_model = Model(input_img, encoder)
X_enc = enc_model.predict(X_dur)
X_test_enc = enc_model.predict(X_test_dur)
X1 = np.concatenate([demo, X_enc], axis=1)
X1_test = np.concatenate([demo_test, X_test_enc], axis=1)

search, cv = search_cv(X1, np.ravel(y), 'rf', 'Classification', 'Bayes')
print(make_msg('rf', 'all-demo', search, cv, 0))

# search, cv = search_cv(X1, np.ravel(y), 'mlp', 'Classification', 'Bayes')
# print(make_msg('mlp', 'all-demo', search, cv, 0))

# search, cv = search_cv(X1, np.ravel(y), 'mlp', 'Classification', 'Bayes')
# print(make_msg('mlp', 'all-demo', search, cv, 0))
