from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import time
import lightgbm as lgb


def verbalise_dataset(train, test):
    print('Train shape:' + str(train.shape))
    print('Test shape:' + str(test.shape))
    print()


def load_file(filepath):

    start_time = time.time()
    df = pd.read_csv(filepath, low_memory=False)
    elapsed_time = time.time() - start_time
    print("Dataset loaded, time elapsed: " + str(elapsed_time))

    return df


def remove_duplicate_col(train, test):

    print('Removing duplicated features')
    remove = []
    cols = train.columns  # list of headers
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            if np.array_equal(train[cols[i]].values, train[cols[j]].values) and cols[j] not in remove:
                remove.append(cols[j])

    train = train.drop(remove, axis=1)
    test = test.drop(remove, axis=1)

    return train, test


# If a column only has a single value, it's std would be zero.
# A column with only one value does not provide any entropy for classification
def remove_constant_col(train, test):

    print('Removing constant features')
    columns = []
    for col in train.columns:
        if train[col].std() == 0:
            columns.append(col)

    train = train.drop(columns, axis=1)
    test = test.drop(columns, axis=1)

    return train, test


if __name__ == '__main__':

    train = load_file('../data/train.csv')  # (76020, 371)
    test = load_file('../data/test.csv')  # (75818, 370)
    verbalise_dataset(train, test)

    train, test = remove_duplicate_col(train, test)
    verbalise_dataset(train, test)  # (76020, 371), (75818, 308)

    train, test = remove_constant_col(train, test)
    verbalise_dataset(train, test)  # (76020, 308), (75818, 307)

    # split data into train and test
    X = train.drop(["TARGET", "ID"], axis=1)
    Y = train['TARGET'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1632)
    print(X_train.shape, X_test.shape)

    d_train = lgb.Dataset(X_train, label=Y_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
    }
    clf = lgb.train(train_set=d_train, params=params)

    Y_pred = clf.predict(X_test)
    print("Score: " + str(roc_auc_score(Y_test, Y_pred)))
