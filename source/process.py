#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    process.py
# @Author:      staceyrivet
# @Time:        3/30/22 10:33 PM
# @IDE:         PyCharm





from utils import *
import numpy as np
from sklearn.preprocessing import StandardScaler






def data_for_cv(data, group_sub_ids, runs_train, runs_test, norm):
    """
    data         : (52 subject data, keys as subject ID for frmi data or labels)
    group_sub_ids: (list of string ID names) either child or teen_plus ids of subjects split on age
    runs_train   : int , (which run are we using for the training data)
    runs_test    : int, (which run are we using for the test data)
    norm         : string, ("RUNS": normalizing separately on each run;
                            "SUBJECT": Normalizing separately by each subject)
    returns      : nd.arrays, Concatenated X data of (time points, x*y*z) x = 79, y = 95, z = 75
                   and Concatenated y labels of (time points,)
    """

    # train and labels
    X = []
    y = []
    # TEST and labels
    Xt = []
    yt = []

    print(f"Normalizing Each based on {norm}...")
    # Get X data from dictionary
    for id_ in group_sub_ids:

        if norm == "RUNS":
            scalar = StandardScaler()
            tr = scalar.fit_transform(data[id_][runs_train])
            X.append(tr)

            scalarT = StandardScaler()
            tst = scalarT.fit_transform(data[id_][runs_test])
            Xt.append(tst)

        elif norm == "SUBJECT":
            print(f"Normalizing Each Subject Data for group {group_sub_ids}")
            scalar = StandardScaler().fit(data[id_][runs_train])
            tr = scalar.transform(data[id_][runs_train])
            X.append(tr)

            tst = scalar.transform(data[id_][runs_test])
            Xt.append(tst)
            
        else:
            X.append(data[id_][runs_train])
            Xt.append(data[id_][runs_test])

        # Get y labels from dictioanry
        y.append(data[f"{id_}_rt_labels"][runs_train])
        yt.append(data[f"{id_}_rt_labels"][runs_test])

    X_c, y_c, X_t, y_t = concat_data((X, y), False, (Xt, yt))

    X_full = np.concatenate((X_c, X_t))
    y_full = np.concatenate((y_c, y_t))

    print("Final X data shape to feed into Time Series Cross Validation", X_full.shape)
    print("Final y data shape to feed into Time Series Cross Validation", y_full.shape)

    return X_full, y_full






def concat_data(train, val, test):
    """
    :param train: Tuple, Array of pre-split training data of shape (time, x*y*z) x = 79, y = 95, z = 75,
                  and Array of y labels where y labels has a shape of (time points,)
    :param val  : Tuple, Array of pre-split validation data of shape (time, x*y*z) x = 79, y = 95, z = 75,
                  and Array of y labels where y labels has a shape of (time points,)
    :param test : Tuple, Array of pre-split test data of shape (time, x*y*z) x = 79, y = 95, z = 75,
                  and Array of y labels where y labels has a shape of (time points,)
    :return     : If validation set is passed in, will return X train, validation, and test data and labels each w/shape
                  (time points, 237979) and (time points,), respectively. Else train and test data will be returned.
    """
    if train:
        x, y = train
        X_c = np.concatenate(np.array(x))
        y_c = np.concatenate(np.array(y))
        print("X train data shape after concantenation", X_c.shape)
        print("y train data shape after concantenation", y_c.shape)

    if test:
        xtest, ytest = test
        X_t = np.concatenate(np.array(xtest))
        y_t = np.concatenate(np.array(ytest))

        print("X test data shape after concantenation", X_t.shape)
        print("y test data shape after concantenation", y_t.shape)

    if val:
        xval, yval = val
        X_v = np.concatenate(np.array(xval))
        y_v = np.concatenate(np.array(yval))

        print("X val data shape after concantenation", X_v.shape)
        print("y val data shape after concantenation", y_v.shape)

        return X_c, y_c, X_v, y_v, X_t, y_t

    return X_c, y_c, X_t, y_t






def transform_data(data, group_sub_ids, runs_train, runs_val, runs_test, norm):
    """
    data.        : (52 subject data, keys as subject ID for frmi data or labels)
    group_sub_ids: (list of string ID names)
    runs_train   : int , (which run are we using for the training data)
    runs_val     : int, which run to use for validation data prediction
    runs_test    : int, (which run are we using for the test data)
    norm         : string, ("RUNS": normalizing separately on each run;
                            "SUBJECT": Normalizing separately by each subject,
                            fitted to the train data of current subject. Can only
                            use when all sets utilize the same subject IDs.)
    :returns     : All data returned as (time points, 237979) and labels (time points,) normalized
    """

    train_id, val_id, test_id = group_sub_ids

    print(f"Normalizing Each based on {norm}...")
    if norm == "RUNS":
        X, y   = scale_data(data, train_id, runs_train, False, norm)
        Xv, yv = scale_data(data, val_id, runs_val, False, norm)
        Xt, yt = scale_data(data, test_id, runs_test, False, norm)

    elif norm == "SUBJECT":
        print("In order to use SUBJECT NORMALIZATION, be sure Train, Val, Test sets")
        print("All have the same subject IDs, and are using data from separate runs.")
        X, y   = scale_data(data, train_id, runs_train, runs_train, norm)
        Xv, yv = scale_data(data, val_id, runs_val, runs_train, norm)
        Xt, yt = scale_data(data, test_id, runs_test, runs_train, norm)

    X_c , y_c, X_v, y_v, X_t, y_t = concat_data((X, y), (Xv, yv), (Xt, yt))


    print( "Final X Train data shape", X_c.shape)
    print( "Final y Train data shape ", y_c.shape)
    print( "Final X Val data shape", X_v.shape)
    print( "Final y Val data shape ", y_v.shape)
    print( "Final X Test data shape", X_t.shape)
    print( "Final y Test data shape ", y_t.shape)

    return X_c , y_c, X_v, y_v, X_t, y_t # Data ready for SVM






def scale_data(data, sub_ids, run, train_run, norm):
    """
    data      : (52 subject data, keys as subject ID for frmi data or labels)
    sub_ids   : (list of string ID names)
    run       : int, (which run are we using for the current data)
    train_run : int , (which run are we using for the training data)
    norm      : string, ("RUNS": normalizing separately on each run;
                        "SUBJECT": Normalizing separately by each subject,
                        fitted to the train data of current subject. Can only
                        use when all sets utilize the same subject IDs.)
    returns   : X and y data normalized based on across all runs for a subject or per subject for a single run
    """

    X = []
    y = []
    scaler = StandardScaler()

    for id_ in sub_ids:
        if norm == "RUNS":
            x = scaler.fit_transform(data[id_][run])
            X.append(x)

        elif norm == "SUBJECT":
            scaler.fit(data[id_][train_run]) # we want our data to be fit on the training data
            x = scaler.transform(data[id_][run]) #transform on the actual data
            X.append(x)
        y.append(data[f"{id_}_rt_labels"][run])

    return X, y





