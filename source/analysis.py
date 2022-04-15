#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    analysis.py
# @Author:      staceyrivet
# @Time:        4/6/22 7:32 PM
# @IDE:         PyCharm


from access_data import s3_upload
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import threshold_img
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import defaultdict






def create_bmaps(clf, X, indices_mask, image):
    """
    :param clf:
    :param X:
    :param indices_mask:
    :param image:
    :return:
    """

    # Create alpha matrix and map weights to support vector indices
    alphas1 = np.zeros((len(X)))
    alphas1[clf.support_] = clf.dual_coef_  # Load the weights corresponding to where support vectors are

    alphas2 = alphas1.reshape(1, -1)
    bmap = np.dot(alphas2, X)
    print("Shape of beta map: ", bmap.shape)

    # Grab the areas not masked out of the brain to recreate the brain using highlighted areas
    bmap2 = np.zeros((79, 95, 79))
    bmap2 = bmap2.reshape(79 * 95 * 79)
    bmap2[indices_mask] = bmap
    bmap2_3 = bmap2.reshape(79, 95, 79, order="F")

    bmap3 = nib.Nifti1Image(bmap2_3, affine=image.affine, header=image.header)

    return bmap3, bmap2_3, alphas1






def get_threshold_image(bmap3, score_percentile, image_intensity):
    """

    :param bmap3:
    :param score_percentile:
    :param image_intensity:
    :return:
    """
    # Two types of strategies can be used from this threshold function
    # Type 1: strategy used will be based on score percentile
    threshold_percentile_img = threshold_img(bmap3, threshold=score_percentile, copy=False)

    # Type 2: threshold strategy used will be based on image intensity
    # Here, threshold value should be within the limits i.e. less than max value.
    threshold_value_img = threshold_img(bmap3, threshold=image_intensity, copy=False)
    return threshold_percentile_img, threshold_value_img






def metrics(clf, X, y, X_v, y_v, X_t, y_t, data_type, runs_id, mask_type):
    """

    :param clf:
    :param X_v:
    :param y_v:
    :param X_t:
    :param y_t:
    :param data_type:
    :param runs_id:
    :param mask_type:
    :param model_type:
    :return:
    """
    metrics_dict = defaultdict(list)
    if X_v != False:
        # Validation metrics
        print("Predicting on Validation set...")
        yval_pred = clf.predict(X_v)
        yval_probs = clf.predict_proba(X_v)[:, 1]
        val_acc = accuracy_score(y_v, yval_pred)
        yval_defunc = clf.decision_function(X_v)
        print("Validation Accuracy:", val_acc)

        #Initialize dict w/ data
        metrics_dict['model'].append(clf)
        metrics_dict['X_train'].append(X)
        metrics_dict["y_train"].append(y)
        metrics_dict['val_dfnc'].append(yval_defunc)
        metrics_dict['val_preds'].append(yval_pred)
        metrics_dict['val_probs'].append(yval_probs)
        metrics_dict['val_acc'].append(val_acc)
        metrics_dict['y_v'].append(y_v)

    # Test Metrics
    print("Predicting on Test set...")
    ytest_pred = clf.predict(X_t)
    ytest_probs = clf.predict_proba(X_t)[:, 1]
    ytest_defunc = clf.decision_function(X_t)
    test_acc = accuracy_score(y_t, ytest_pred)
    print("Test Accuracy:", test_acc)

    #Store metrics in dictionary
    metrics_dict['model'].append(clf)
    metrics_dict['X_train'].append(X)
    metrics_dict["y_train"].append(y)
    metrics_dict['test_preds'].append(ytest_pred)
    metrics_dict['test_probs'].append(ytest_probs)
    metrics_dict['test_acc'].append(test_acc)
    metrics_dict['test_dfunc'].append(ytest_defunc)
    metrics_dict['y_t'].append(y_t)

    model_name = f"{data_type}_{runs_id}_{mask_type}_model_metrics"
    # Save metrics and model
    s3_upload(metrics_dict, f"metrics/group_svm/{mask_type}/%s.pkl" % model_name, 'pickle')


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

        s3_upload(df, f"metrics/group_svm/{mask_type}/{data_type}_model_{runs_id}_{mask_type}_{report}.csv", "csv")
        print(f"Classification report for {mask_type} {report}")
        print(classification_report(y_v, yval_pred))

    return metrics_dict
