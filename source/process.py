#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    process.py
# @Author:      staceyrivet
# @Time:        3/30/22 10:33 PM
# @IDE:         PyCharm




from access_data import access_load_data
import tqdm
from nilearn.signal import clean
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import defaultdict






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
    if len(runs_train) > 0:
        if len(group_sub_ids) == 2:
            X, y = scale_data(data, group_sub_ids[0], runs_train, runs_train, norm)
            Xt, yt = scale_data(data, group_sub_ids[1], runs_train, runs_train, norm)
    else:
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
                print("No extra normalization...")

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

    if test:
        xtest, ytest = test
        X_t = np.concatenate(np.array(xtest))
        y_t = np.concatenate(np.array(ytest))

    if val:
        xval, yval = val
        X_v = np.concatenate(np.array(xval))
        y_v = np.concatenate(np.array(yval))

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
    if norm == "RUNS":
        print(f"Normalizing Each based on {norm}...")
    elif norm == "Detrend_Znorm":
        print("Data already detrended and znorm scaled by columns per run...")

    if runs_val != False:
        print(f"Running with a validation set...")
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

        elif norm == 'Detrend_Znorm':
            X, y   = scale_data(data, train_id, runs_train, runs_train, norm)
            Xv, yv = scale_data(data, val_id, runs_val, runs_train, norm)
            Xt, yt = scale_data(data, test_id, runs_test, runs_train, norm)

        X_c, y_c, X_v, y_v, X_t, y_t = concat_data((X, y), (Xv, yv), (Xt, yt))

        print("Final X Train data shape", X_c.shape)
        print("Final y Train data shape ", y_c.shape)
        print("Final X Val data shape", X_v.shape)
        print("Final y Val data shape ", y_v.shape)
        print("Final X Test data shape", X_t.shape)
        print("Final y Test data shape ", y_t.shape)

        return X_c, y_c, X_v, y_v, X_t, y_t  # Data ready for SVM

    else:
        print("Crossvalidation already completed, do not need validation set.")
        if norm == "RUNS":
            X, y   = scale_data(data, train_id, runs_train, False, norm)
            Xt, yt = scale_data(data, test_id, runs_test, False, norm)

        elif norm == "SUBJECT":
            print("In order to use SUBJECT NORMALIZATION, be sure Train, Val, Test sets")
            print("All have the same subject IDs, and are using data from separate runs.")
            X, y   = scale_data(data, train_id, runs_train, runs_train, norm)
            Xt, yt = scale_data(data, test_id, runs_test, runs_train, norm)

        elif norm == 'Detrend_Znorm':
            X, y   = scale_data(data, train_id, runs_train, runs_train, norm)
            Xt, yt = scale_data(data, test_id, runs_test, runs_train, norm)

        X_c, y_c, X_t, y_t = concat_data((X, y), False, (Xt, yt))


    print( "Final X Train data shape", X_c.shape)
    print( "Final y Train data shape ", y_c.shape)
    print( "Final X Test data shape", X_t.shape)
    print( "Final y Test data shape ", y_t.shape)

    return X_c , y_c, X_t, y_t # Data ready for SVM






def scale_data(data, sub_ids, runs, train_run, norm):
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
            if len(runs) > 0:
                print("znormalizing the data using standard scaler...")
                for run in runs:
                    x = scaler.fit_transform(data[id_][run])
                    X.append(x)
                    y.append(data[f"{id_}_rt_labels"][run])
            else:
                x = scaler.fit_transform(data[id_][runs])
                X.append(x)
                y.append(data[f"{id_}_rt_labels"][runs])

        elif norm == "SUBJECT":
            scaler.fit(data[id_][train_run])  # we want our data to be fit on the training data
            x = scaler.transform(data[id_][runs])  # transform on the actual data
            X.append(x)
            y.append(data[f"{id_}_rt_labels"][runs])

        elif norm == 'Detrend_Znorm':
            if len(runs) > 0:
                print("Detrending and znormalizing the data...")
                for run in runs:
                    x = clean(data[id_][run],
                              standardize='zscore',
                              detrend=True,
                              filter=False,
                              standardize_confounds=False
                              )
                    X.append(x)
                    y.append(data[f"{id_}_rt_labels"][run])

            else:
                clean(data[id_][runs],
                      standardize='zscore',
                      detrend=True,
                      filter=False,
                      standardize_confounds=False
                      )
                X.append(x)
                y.append(data[f"{id_}_rt_labels"][runs])



    return X, y





def create_mask(mask_data_filepath, mask_type='mask'):
    """
    mask_data_filepath: Takes in filepaths
    mask_type: and mask type
    returns: a numpy-compatible mask

    """
    mask_type_dict = access_load_data(mask_data_filepath, True)
    np_array_mask = mask_type_dict[mask_type]
    np_compatible_mask = np.ma.make_mask(np_array_mask).reshape(79 * 95 * 79, order='F')

    return np_compatible_mask






def labels_mask_binary(label_data_path, label_type='rt_labels'):
    """

    """
    # Get Mask data
    label_data_dict = access_load_data(label_data_path, True)
    labels = label_data_dict[label_type]
    mask_labels_indices = np.where(labels != 9999)
    mask_labels_indices = mask_labels_indices[0]

    # Get Binary Labels
    label_mask = np.all(labels != 9999, axis=1)
    binary_labels = labels.reshape(-1)[label_mask]

    return mask_labels_indices, binary_labels






def masking_data(subject, mask, mask_labels, binary_labels):
    """


    """

    arr = []
    label_arr = []
    for i in tqdm.tqdm(range(4)):
        user_key = 'run_0' + str(i + 1) + '_vec'
        array = subject[user_key]
        array_masked = array[:, mask]
        array_masked = array_masked[mask_labels]

        arr.append(array_masked)
        label_arr.append(binary_labels)

    return arr, label_arr






def masked_data_n_labels(mask_type, label_type, path_dict, m_path_ind, l_path_ind):
    """
    mask_type: String for type of mask we want
    label_type: String for which labels we want
    path_dict: dictionary that contains all the paths for data in AWS
    norm: To normalize on PSC True or False
    m_path_ind: the index to get either masks.mat or roi_masks.mat , 0 or 1 respectively
    l_path_ind: index to get the data from labels in dictionary
    Returns: a dictionary of subject masked data and the cleaned labels
    """
    # Define variable to return
    user_data_dict = defaultdict(list)

    # Mask path from S3
    mask_data_filepath = path_dict['mask_data'][m_path_ind]
    mask_type = mask_type  # 'mask'
    mask = create_mask(mask_data_filepath, mask_type)

    # Label path from S3
    label_data_path = path_dict['labels'][l_path_ind]
    label_type = label_type  # "rt_labels"

    # Returns two values, mask_labels_indices and binary_labels
    mask_labels_indices, binary_labels = labels_mask_binary(label_data_path, label_type)

    # Loop through all subjects
    for ind, val in tqdm.tqdm(enumerate(path_dict['subject_ID'])):
        sub_id = val
        sub_path = path_dict['subject_data'][ind]
        subject = access_load_data(sub_path, True)
        user_data_dict[sub_id], bi_lb = masking_data(subject, mask, mask_labels_indices, binary_labels)
        user_data_dict[f"{sub_id}_{label_type}"] = bi_lb

    return user_data_dict