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
    alphas1[clf.support_] = clf.dual_coef_ #Load the weights corresponding to where support vectors are

    alphas2 = alphas1.reshape(1,-1)
    bmap = np.dot(alphas2, X)
    print("Shape of beta map: ", bmap.shape)

    # Grab the areas not masked out of the brain to recreate the brain using highlighted areas
    bmap2 = np.zeros((79,95,79))
    bmap2 = bmap2.reshape(79*95*79)
    bmap2[indices_mask] = bmap
    bmap2_3 = bmap2.reshape(79,95,79, order = "F")

    bmap3 = nib.Nifti1Image(bmap2_3, affine = image.affine, header = image.header)

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
    threshold_percentile_img = threshold_img(bmap3, threshold = score_percentile, copy = False)


    # Type 2: threshold strategy used will be based on image intensity
    # Here, threshold value should be within the limits i.e. less than max value.
    threshold_value_img = threshold_img(bmap3, threshold = image_intensity, copy = False)
    return threshold_percentile_img, threshold_value_img






def metrics(clf, X_v, y_v, X_t, y_t, data_type, runs_id, mask_type):
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
    model_type = 0

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

        s3_upload(df, f"metrics/group_svm/{data_type}_model_{runs_id}_{mask_type}_{report}.csv", "csv")
        print(f"Classification report for {mask_type} {report}")
        print(classification_report(y_v, yval_pred))

    return True








