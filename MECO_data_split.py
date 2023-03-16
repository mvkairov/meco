import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


def concat_MECO_langs(*langs, path_to_data='Datasets/DataToUse/'):
    if langs == ['all']:
        langs = lang_list
    return pd.concat([pd.read_csv(f'{path_to_data}{lang}_fix_demo.csv') for lang in langs])


class MECODataSplit:
    def __init__(self, lang_data, model='Classification'):
        self.data = lang_data
        self.data = self.data.astype({
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
        })
        self.model = model

        self.fix_cols = ['Fix_X', 'Fix_Y', 'Fix_Duration']
        self.demo_cols = ['motiv', 'IQ', 'Age', 'Sex']

    def get_X_y(self, labels, include_cols):
        X = []
        y = []
        combinations = np.array(np.meshgrid(*labels.values())).T.reshape(-1, len(labels))

        for values in combinations:
            condition_query = ""
            for col, value in zip(labels.keys(), values):
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    condition_query += f"({col} == {value}) & "
                else:
                    condition_query += f"({col} == '{value}') & "
            cur_rows = self.data.query(condition_query[:-2])
            if not cur_rows.empty:
                X.append(cur_rows[include_cols])
                if self.model == 'Classification':
                    y.append(cur_rows['Target_Label'].iloc[0])
                else:
                    y.append(cur_rows['Target_Ave'].iloc[0])

        return X, y

    def resample_X_y(self, X, y, resample_type=None, series_length=None):
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
            include_cols = self.fix_cols
        elif include_cols == 'demo':
            include_cols = self.fix_cols + self.demo_cols

        X, y = self.get_X_y(labels=train_labels, include_cols=include_cols)
        if resample is not None:
            X, y = self.resample_X_y(X, y, resample_type=resample, series_length=series_length)

        if test_size == 0:
            return X, y
        else:
            if test_labels is not None:
                X_test, y_test = self.get_X_y(labels=test_labels, include_cols=include_cols)
                if resample is not None:
                    X_test, y_test = self.resample_X_y(X_test, y_test, resample_type=resample,
                                                       series_length=series_length)
            else:
                X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X, X_test, y, y_test
