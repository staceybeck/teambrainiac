#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    analysis.py
# @Author:      staceyrivet
# @Time:        4/6/22 7:32 PM
# @IDE:         PyCharm


#from access_data import s3_upload, open_pickle, load_mask_indices, access_load_data
from access_data import *
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import threshold_img
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import defaultdict






def get_roi_bmaps(previous_bmap_data, indices_mask, affine_image):
    """

    :param previous_bmap_data:
    :param indices_mask:
    :param affine_image:
    :return:
    """
    bmap2 = previous_bmap_data.reshape(79 * 95 * 79)
    bmap2_3 = np.zeros((79, 95, 79))
    bmap2_3 = bmap2_3.reshape(79 * 95 * 79)
    bmap2_3[indices_mask] = bmap2[indices_mask]
    bmap2_3d = bmap2_3.reshape(79, 95, 79, order='F')
    bmap3d = nib.Nifti1Image(bmap2_3d,
                             affine=affine_image.affine,
                             header=affine_image.header
                             )
    return bmap3d






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

    return bmap3, bmap2_3, alphas1, bmap






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






def metrics(clf, X, y, X_v, y_v, X_t, y_t, data_type, runs_id, mask_type, m_path_ind):
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
    model_dict = defaultdict(list)

    if X_v != False:
        # Validation metrics
        print("Predicting on Validation set...")
        yval_pred = clf.predict(X_v)
        yval_probs = clf.predict_proba(X_v)[:, 1]
        val_acc = accuracy_score(y_v, yval_pred)
        yval_defunc = clf.decision_function(X_v)
        print("Validation Accuracy:", val_acc)

        #Initialize dict w/ data
        model_dict['model'].append(clf)
        model_dict['X_train'].append(X)
        metrics_dict["y_train"].append(y)
        metrics_dict['val_dfnc'].append(yval_defunc.astype(np.float32))
        metrics_dict['val_preds'].append(yval_pred)
        metrics_dict['val_probs'].append(yval_probs.astype(np.float32))
        metrics_dict['val_acc'].append(val_acc)
        metrics_dict['y_v'].append(y_v)

    # Test Metrics
    print("Predicting on Test set...")
    ytest_pred = clf.predict(X_t)
    #ytest_probs = clf.predict_proba(X_t)[:, 1]
    #ytest_defunc = clf.decision_function(X_t)
    test_acc = accuracy_score(y_t, ytest_pred)
    #print("Test Accuracy:", test_acc)

    # Get Bmaps
    # Load in the mask indices to create bmaps
    # open path dictionary file to get subject ids
    print("opening data dictionary to make bmaps...")
    dict_path = "data/data_path_dictionary.pkl"
    data_path_dict = open_pickle(dict_path)
    indices_mask = load_mask_indices(data_path_dict, mask_type, m_path_ind)

    print("creating bmaps...")
    #affine_image = access_load_data('w3rtprun_01.nii', False)
    #bmap3, bmap2_3, alphas1, bmap = create_bmaps(clf, X, indices_mask, affine_image)
    
    
    alphas1 = np.zeros((len(X)))
    alphas1[clf.support_] = clf.dual_coef_  # Load the weights corresponding to where support vectors are

    alphas2 = alphas1.reshape(1, -1)
    bmap = np.dot(alphas2, X)
    print("Shape of beta map: ", bmap.shape)

    # Grab the areas not masked out of the brain to recreate the brain using highlighted areas
    bmap2 = np.zeros((79, 95, 79))
    bmap2 = bmap2.reshape(79 * 95 * 79)
    bmap2[indices_mask] = bmap

    
    
    #Store metrics in dictionary
    #model_dict['model'].append(clf)
    #model_dict['X_train'].append(X)

    #metrics_dict["bmap3"].append(bmap3)
    #metrics_dict['bmap2_3'].append(bmap2_3)
    #metrics_dict['alphas1'].append(alphas1)
    #metrics_dict['bmap'].append(bmap)
    metrics_dict['bmap2'].append(bmap2)

    #metrics_dict["y_train"].append(y)
    metrics_dict['test_preds'].append(ytest_pred)
    #metrics_dict['test_probs'].append(ytest_probs.astype(np.float32))
    metrics_dict['test_acc'].append(test_acc)
    #metrics_dict['test_dfunc'].append(ytest_defunc.astype(np.float32))
    #metrics_dict['y_t'].append(y_t)

    # File paths
    metrics_name = f"BMAP_{data_type}_{runs_id}_{mask_type}_metrics"
    #model_name = f"{data_type}_{runs_id}_{mask_type}_X_model"

    # Save metrics and model
    #s3_upload(metrics_dict, f"metrics/group_svm/{mask_type}/%s.pkl" % metrics_name, 'pickle')
    # Save model
    #s3_upload(model_dict, "models/group/%s.pkl" % model_name, 'pickle')

    # Save locally
    #df = pd.DataFrame(metrics_name)
    #df.to_pickle(f'/content/gdrive/MyDrive/{data_type}_{mask_type}_df.pkl', protocol=4)
    #df = 0
    #print("Uploading to cloud and local computer complete")

    """
    # Save metrics for individual masks
    if X_v != False:
        type_report = ['validation_classreport', 'test_classreport']
    else:
        type_report = ['test_classreport']

    for report in type_report:
        if report == 'validation_classreport':
            class_report = classification_report(y_v, yval_pred, output_dict=True)
            class_report.update({"accuracy": {"precision": None, "recall": None,
                                              "f1-score": class_report["accuracy"],
                                              "support": class_report['macro avg']['support']}})
            df = pd.DataFrame(class_report).T
            print(f"Classification report for {mask_type} {report}")
            print(classification_report(y_v, yval_pred))

        elif report == 'test_classreport':
            class_report = classification_report(y_t, ytest_pred, output_dict=True)
            class_report.update({"accuracy": {"precision": None, "recall": None,
                                              "f1-score": class_report["accuracy"],
                                              "support": class_report['macro avg']['support']}})
            df = pd.DataFrame(class_report).T
            print(f"Classification report for {mask_type} {report}")
            print(classification_report(y_t, ytest_pred))

        s3_upload(df, f"metrics/group_svm/{mask_type}/{data_type}_model_{runs_id}_{mask_type}_{report}.csv", "csv")
    """
    return metrics_dict
