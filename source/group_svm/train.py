#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    train.py
# @Author:      staceyrivet
# @Time:        4/6/22 7:30 PM
# @IDE:         PyCharm







from process import transform_data
from sklearn.svm import SVC
from collections import defaultdict
from analysis import metrics







def run_grp_svm_model(data, mask_type, group_sub_ids, runs_train, runs_val, runs_test, norm, data_type, m_path_ind):
    """
    Classifier set up for two different groups based on current best parameters
    Trains the model
    calls a metric function to get a dictionary of metric values
    save model as a dictionary to store locally (since having issues with pickles)


    :param data:            Masked and filter labelled data
    :param mask_type:       (string) mask label we want
    :param group_sub_ids:   array of string IDs
    :param runs_train:      array of int/ints
    :param runs_val:        if True, array of int/ints, else False
    :param runs_test:       array of int/ints
    :param norm:            (string) normalization label to apply for processing
    :param data_type:       (string) group label we specify (young adult versus adolscent)
    :param m_path_ind:      (int) binary 0 or 1 for mask path in AWS
    :return:                Dictionary containing the model, dictionary containing all the metric data
    """

    model_dict = defaultdict(list)

    if runs_val != False:
        X, y, X_v, y_v, X_t, y_t = transform_data(data, group_sub_ids, runs_train, runs_val, runs_test, norm)

        runs_id = [i + 1 for i in runs_train]

        if data_type == "AD_detrend":
            print(f"Train on {data_type}")
            clf = SVC(C=5.0, #10
                      class_weight='balanced',
                      max_iter=1000,
                      random_state=42,
                      probability=True,
                      gamma='scale' #For adolescent
                      )
        else:
            print(f"Train on {data_type}")
            clf = SVC(C=10.0,
                      class_weight='balanced',
                      max_iter=1000,
                      random_state=42,
                      probability = True,
                      gamma = 'auto'
                      )
        print(f"Fitting the model for {mask_type}...")
        clf.fit(X, y)

        model_dict['model'] = clf

        # Calculate metrics
        metrics_data = metrics(clf, X, y, X_v, y_v, X_t, y_t, data_type, runs_id, mask_type, m_path_ind)
    else:
        X, y, X_t, y_t = transform_data(data, group_sub_ids, runs_train, runs_val, runs_test, norm)

        runs_id = [i + 1 for i in runs_train]

        if data_type == "AD_detrend":
            print(f"Train on {data_type}")
            clf = SVC(C=5.0, #10
                      class_weight='balanced',
                      max_iter=1000,
                      random_state=42,
                      probability=True,
                      gamma='scale' #For adolescent
                      )
        else:
            print(f"Train on {data_type}")
            clf = SVC(C=10.0,
                      class_weight='balanced',
                      max_iter=1000,
                      random_state=42,
                      probability=True,
                      gamma = 'auto' # for young adult
                      )
        print(f"Fitting the model for {mask_type} on Train and then will Test...")
        clf.fit(X, y)

        model_dict['model'] = clf

        # Calculate metrics
        metrics_data = metrics(clf, X, y, False, False, X_t, y_t, data_type, runs_id, mask_type, m_path_ind)


    # Return metrics data
    return model_dict, metrics_data