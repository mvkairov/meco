import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

'''

requirements:

- split data by values of all specified columns
- return uneven or resampled data (stretching, truncation or other methods)
- resample into combined fixations on each word
- split data into train/test: randomly or test specific elements (Text_ID, SubjectID or others /
                                                                  n or a fraction of elements from specified columns)
- return data with or without fixation coordinates or demographics
- graph data: autocorrelation, correlation between languages...

'''

lang_list = ['du', 'ee', 'fi', 'ge', 'gr', 'he', 'it', 'no', 'ru', 'sp', 'it']
fix_cols = ['Fix_X', 'Fix_Y', 'Fix_Duration']
demo_cols = ['motiv', 'IQ', 'Age', 'Sex']
data_types = {
    "Text_ID": int,
    "Fix_X": int,
    "Fix_Y": int,
    "Fix_Duration": int,
    "Word_Number": int,
    "Sentence": str,
    "Language": str,
    "SubjectID": str,
    "L2_spelling_skill": float,
    "L2_vocabulary_size": float,
    "vocab.t2.5": float,
    "L2_lexical_skill": float,
    "TOWRE_word": float,
    "TOWRE_nonword": float,
    "motiv": float,
    "IQ": float,
    "Age": int,
    "Sex": int,
    "Target_Ave": float,
    "Target_Label": int
}


def concat_MECO_langs(langs, path_to_data='../Datasets/DataToUse/'):
    if langs == ['all']:
        langs = lang_list
    return pd.concat([pd.read_csv(f'{path_to_data}{lang}_fix_demo.csv') for lang in langs])


def split_into_rows(data, cols='fix+demo', target='Target_Label', with_values_only=None, shuffle_data=False):
    if cols == 'fix':
        cols = fix_cols
    elif cols == 'demo':
        cols = demo_cols
    elif cols == 'fix+demo':
        cols = fix_cols + demo_cols

    data = data.astype(data_types)
    if with_values_only is not None:
        for col, values in with_values_only.items():
            values = [data_types[col](value) for value in values]
            data = data[data[col].isin(values)]
    if cols == 'demo':
        data.drop_duplicates(subset=cols+[target], keep='first', inplace=True)

    X, y = data[cols], data[target]
    if shuffle_data:
        X, y = shuffle(X, y)
    return X, y


def make_dataset(langs, cols='fix', target='Target_Label', params=None,
                 cv_col='SubjectID', path_to_data='Datasets/DataToUse/'):
    data = concat_MECO_langs(langs, path_to_data=path_to_data)
    X, y = split_into_rows(data, cols=cols, target=target, with_values_only=params)

    if cv_col is not None:
        keys = data[cv_col].unique()
        cv_dict = {key: num for num, key in enumerate(keys)}
        cv_col = np.vectorize(cv_dict.get)(data[cv_col].to_numpy())
    return X, y, cv_col


def resample_Xy(X, y, resample_type=None, series_length=None):
    if resample_type == 'truncate':
        remove_indices = []
        for i in range(len(X)):
            if len(X[i]) >= series_length:
                X[i] = X[i][:series_length]
            else:
                remove_indices.append(i)
        for i in sorted(remove_indices, reverse=True):
            if i < len(X):
                X.pop(i)
                y.pop(i)
    return np.array(X), np.array(y)


class MECODataSplit:
    def __init__(self, langs, target_row=None):
        if target_row is None:
            target_row = 'Target_Label'
        self.target_row = target_row

        self.data = concat_MECO_langs(langs)
        self.data = self.data.astype(data_types)

    def get_Xy(self, labels, include_cols):
        X, y = [], []
        combinations = np.array(np.meshgrid(*labels.values())).T.reshape(-1, len(labels))

        for values in combinations:
            condition_query = ''
            for col, value in zip(labels.keys(), values):
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    condition_query += f'({col} == {value}) & '
                else:
                    value = value.replace("'", '`')
                    condition_query += f'({col} == "{value}") & '
            cur_rows = self.data.query(condition_query[:-2])
            if not cur_rows.empty:
                X.append(cur_rows[include_cols])
                y.append(cur_rows[[self.target_row]].iloc[0])
        return X, y

    def split_by_unique_values(self, split_cols, include_cols='fix',
                               train_on_values=None, test_on_values=None,
                               fix_threshold=0, resample=None, series_length=None,
                               test_size=0.15, random_state=42):
        self.data = self.data[self.data['Fix_Duration'] >= fix_threshold]
        if train_on_values is not None:
            train_labels = train_on_values
        else:
            train_labels = dict(zip(split_cols, [pd.unique(self.data[col]) for col in split_cols]))
        test_labels = test_on_values

        if include_cols == 'fix':
            include_cols = fix_cols
        elif include_cols == 'fix+demo':
            include_cols = fix_cols + demo_cols

        X, y = self.get_Xy(labels=train_labels, include_cols=include_cols)
        if resample is not None:
            X, y = resample_Xy(X, y, resample_type=resample, series_length=series_length)

        if test_size == 0:
            return X, y
        else:
            if test_labels is not None:
                X_test, y_test = self.get_Xy(labels=test_labels, include_cols=include_cols)
                if resample is not None:
                    X_test, y_test = resample_Xy(X_test, y_test, resample_type=resample, series_length=series_length)
            else:
                X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X, X_test, y, y_test
