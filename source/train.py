#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    train.py.py
# @Author:      staceyrivet
# @Time:        4/6/22 7:30 PM
# @IDE:         PyCharm







from utils import *
from process import *
from cross_validation import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report







def run_grp_svm_model(data, mask_type, group_sub_ids, runs_train, runs_val, runs_test, norm, svm_type):
    """

    :param data:
    :param mask_type:
    :param group_sub_ids:
    :param runs_train:
    :param runs_val:
    :param runs_test:
    :param norm:
    :param svm_type:
    :return:
    """

    X, y, X_v, y_v, X_t, y_t = transform_data(data, group_sub_ids, runs_train, runs_val, runs_test, norm)

    runs_id = [i + 1 for i in runs_train]
    model_dict = defaultdict(list)

    model_name = f"{svm_type}_{runs_id}_{mask_type}_X_y_model"
    clf = SVC(C=5.0, class_weight='balanced', max_iter=1000, random_state=42)  # probability = True
    print(f"Fitting the model for {mask_type}...")
    clf.fit(X, y)
    model_dict['model'].append(clf)
    model_dict['X_train'].append(X)
    model_dict["y_train"].append(y)

    s3_upload(model_dict, "models/group/%s.pkl" % model_name, 'pickle')

    print("Predicting on Validation set...")
    yval_pred = clf.predict(X_v)
    val_acc = accuracy_score(y_v, yval_pred)
    print("Validation Accuracy:", val_acc)

    print("Predicting on Test set...")
    ytest_pred = clf.predict(X_t)
    test_acc = accuracy_score(y_t, ytest_pred)
    print("Test Accuracy:", test_acc)

    # Save metrics for individual masks
    type_report = ['validation_classreport', 'test_classreport']
    for report in type_report:
        if report == 'validation_classreport':
            class_report = classification_report(y_v, yval_pred, output_dict=True)
            class_report.update({"accuracy": {"precision": None, "recall": None,
                                              "f1-score": class_report["accuracy"],
                                              "support": class_report['macro avg']['support']}})
            df = pd.DataFrame(class_report).T

        elif report == 'test_classreport':
            class_report = classification_report(y_t, ytest_pred, output_dict=True)
            class_report.update({"accuracy": {"precision": None, "recall": None,
                                              "f1-score": class_report["accuracy"],
                                              "support": class_report['macro avg']['support']}})
            df = pd.DataFrame(class_report).T

        s3_upload(df, f"metrics/group_svm/{svm_type}_{runs_id}_{mask_type}_{report}.csv", "csv")
        print(f"Classification report for {mask_type} {report}")
        print(classification_report(y_v, yval_pred))

    # Return X train and clf for visualization
    return model_dict