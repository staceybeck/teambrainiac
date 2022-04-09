#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    group_svm.py
# @Author:      staceyrivet
# @Time:        4/8/22 12:52 PM
# @IDE:         PyCharm





import nibabel as nib
import os
import scipy.io
#import matplotlib.pyplot as plt
#from ipywidgets import interact
import pickle
from path_config import mat_path
import boto3
import tempfile
from collections import defaultdict
import tqdm
#from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean
from botocore.exceptions import ClientError
from utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import *
from process import *
from cross_validation import *
import numpy as np
import nibabel as nib
from nilearn.image import threshold_img
from utils import *
from process import *
from cross_validation import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report






def data_to_nib(path):
    """
    using Nibabel open NiFTI files

    :param nifti data path:
    :return loads nifti data:
    """

    filename = os.path.join(path)
    return nib.load(filename)






def load_mat(path):
    """

    :param .mat data path
    uses scipy to convert access to mat file in python
    :return mat data

    """
    mat_file = scipy.io.loadmat(path)
    return mat_file






def open_pickle(file_path):
    """
    :param file path for dictionary
    :return dictionary:
    """
    f = open(file_path, "rb")
    dictionary = pickle.load(f)
    f.close()

    return dictionary






def access_load_data(obj, bool_mat):
    """
    :param data_file: file path to data. For example can be pickle file containing a dictionary
                      or csv file path to load as a dataframe, e.g. file_names_dict['subject_data'][0]
                      or nifti file path to load as nifti object
                      or load a matlab file if bool_mat == TRUE

    :param bool_mat:  if true will run load_mat() and return .mat file

    :return        :  will open matlab data if bool_mat == True,
                      pickle file
                      csv file as a dataframe
                      nifti file
    """


    # Connect to AWS client
    _, bucket_, client = access_aws()

    # Create a temporary file
    temp = tempfile.NamedTemporaryFile()

    # Download data in temp file
    client.download_file(bucket_, obj, temp.name)

    if bool_mat == True:
        data = load_mat(temp.name)
    else:
        if '.pkl' in obj:
            data = open_pickle(temp.name)
        elif '.csv' in obj:
            data = pd.read_csv(temp.name)
        elif '.nii' in obj:
            temp_f = 'data/temp.nii'
            client.download_file(bucket_, obj, temp_f)
            data = data_to_nib(temp_f)

    temp.close()

    return data






def s3_upload(data, object_name, data_type):
    """Upload a file to an S3 bucket
    :param data: our data to upload
    :param data_type: type of data file we are creating
    :param object_name: S3 object name. If not specified then name of temp.name is used
    :return: True if file was uploaded, else False
    """

    # Upload the file
    # Connect to AWS client
    pubkey = mat_path['ACCESS_KEY']
    seckey = mat_path['SECRET_KEY']
    client = boto3.client('s3', aws_access_key_id=pubkey, aws_secret_access_key=seckey)
    s3 = boto3.resource('s3', aws_access_key_id=pubkey, aws_secret_access_key=seckey)

    # Grab bucket name
    bucket = s3.Bucket('teambrainiac')
    bucket_name = bucket.name  # 'teambrainiac'
    try:

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            if data_type == "pickle":
                pickle.dump(data, temp)

            elif data_type == "numpy":
                np.save(temp, data)
                _ = temp.seek(0)

            elif data_type == "csv":
                data.to_csv(temp, index=False)

            client.upload_file(temp.name, bucket_name, object_name)
            temp.close()
            print(f"upload complete for {object_name}")

        if data_type == "nifti":
            tempf = 'data/upload_temp.nii'
            nib.save(data, tempf)
            client.upload_file(tempf, bucket_name, object_name)
            print(f"upload complete for {object_name}")

    except ClientError as e:
        logging.error(e)
        return False

    return True






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
    binary_labels

    return mask_labels_indices, binary_labels






def masking_data(subject, mask, mask_labels, binary_labels, do_norm):
    """


    """

    arr = []
    label_arr = []
    for i in tqdm.tqdm(range(4)):
        user_key = 'run_0' + str(i + 1) + '_vec'
        array = subject[user_key]
        array_masked = array[:, mask]
        array_masked = array_masked[mask_labels]

        # Percent Signal Change normalization
        if do_norm:
            array_masked = clean(array_masked, standardize='psc')

        arr.append(array_masked)
        label_arr.append(binary_labels)

    return arr, label_arr






def masked_data_n_labels(mask_type, label_type, path_dict, do_norm, m_path_ind, l_path_ind):
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
        user_data_dict[sub_id], bi_lb = masking_data(subject, mask, mask_labels_indices, binary_labels, do_norm)
        user_data_dict[f"{sub_id}_{label_type}"] = bi_lb

    return user_data_dict






def plot_alphas(alphas, alpha_labels, title):
  """

  :param alphas:
  :param alpha_labels:
  :param title:
  :return:
  """
  fig, ax = plt.subplots(1, 1, figsize=(15, 5))
  ax.plot(alphas, lw=3, label='scaled voxel tc')
  ax.plot(alpha_labels, lw=3, label='predictor tc')
  # ax.set_xlim(0, acq_num-1)
  ax.set_xlabel('time [volumes]', fontsize=20)
  ax.tick_params(labelsize=12)
  ax.set_title(title)
  ax.legend()
  plt.show()






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

    print(f"Normalizing Each based on {norm}...")
    if norm == "RUNS":
        X, y = scale_data(data, train_id, runs_train, False, norm)
        Xv, yv = scale_data(data, val_id, runs_val, False, norm)
        Xt, yt = scale_data(data, test_id, runs_test, False, norm)

    elif norm == "SUBJECT":
        print("In order to use SUBJECT NORMALIZATION, be sure Train, Val, Test sets")
        print("All have the same subject IDs, and are using data from separate runs.")
        X, y = scale_data(data, train_id, runs_train, runs_train, norm)
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
            if len(run) > 0:
                for runs in run:
                    x = scaler.fit_transform(data[id_][runs])
                    X.append(x)
                    y.append(data[f"{id_}_rt_labels"][runs])
            else:
                x = scaler.fit_transform(data[id_][run])
                X.append(x)
                y.append(data[f"{id_}_rt_labels"][run])

        elif norm == "SUBJECT":
            scaler.fit(data[id_][train_run])  # we want our data to be fit on the training data
            x = scaler.transform(data[id_][run])  # transform on the actual data
            X.append(x)
            y.append(data[f"{id_}_rt_labels"][run])

    return X, y





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





from utils import *
from process import *
from cross_validation import *
import numpy as np
import nibabel as nib
from nilearn.image import threshold_img





def create_bmaps(clf, X, indices_mask, image):

    """


    :param clf:
    :param X:
    :param indices_mask:
    :param image:
    :return:
    """

    # Create alpha matrix and map weights to support vector indices
    alphas = np.zeros((len(X)))
    alphas[clf.support_] = clf.dual_coef_ #Load the weights corresponding to where support vectors are

    alphas = alphas.reshape(1,-1)
    bmap = np.dot(alphas, X)
    print("Shape of beta map: ", bmap.shape)

    # Grab the areas not masked out of the brain to recreate the brain using highlighted areas
    bmap2 = np.zeros((79,95,79))
    bmap2[indices_mask] = bmap

    bmap3 = nib.Nifti1Image(bmap2, affine = image.affine, header = image.header)

    return bmap3






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






def metrics(clf, X_v, y_v, X_t, y_t, svm_type, runs_id, mask_type):
    """

    :param clf:
    :param X_v:
    :param y_v:
    :param X_t:
    :param y_t:
    :param svm_type:
    :param runs_id:
    :param mask_type:
    :return:
    """


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