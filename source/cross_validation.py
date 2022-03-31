#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    cross_validation.py
# @Author:      staceyrivet
# @Time:        3/30/22 10:49 PM
# @IDE:         PyCharm






from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning
import warnings
import tqdm






def time_series_cv(X, y, max_train, test_size, splits):

    """
    :param X        : nd.array shape (time points, 237979)
    :param y        : nd.array shape (time points, )
    :param max_train: (int) max size to allow training set for splits
    :param test_size: (int) max size to allow test split
    :param splits   : (int) number of times to split for cross validation
    :return         : prints accuracy scores and mean accuracy score over all cross validation runs
    """

    tscv = TimeSeriesSplit(max_train_size=max_train, n_splits=splits, test_size=test_size)
    accuracy_ = []
    it = 0

    for train_index, test_index in tqdm.tqdm(tscv.split(X)):
        it += 1
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf = SVC(C=5.0, class_weight='balanced', max_iter=1000, random_state=42)
            print(f"Fitting Classifier for iteration number {it}")
            clf.fit(X_train, y_train)

        print("Predicting...")
        y_pred = clf.predict(X_test)

        # Model Accuracy
        score = accuracy_score(y_test, y_pred)
        print(f"Cross Validation Split {it} Accuracy score:", score)
        accuracy_.append(score)

    print("Mean Accuracy: {}".format(np.mean(accuracy_)))
