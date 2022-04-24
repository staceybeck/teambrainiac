#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    access_data.py
# @Author:      staceyrivet, benmerrill, marysoules
# @Time:        4/8/22 1:51 PM
# @IDE:         PyCharm




import re
import scipy.io
import os
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
from path_config import mat_path
import boto3
import tempfile
from botocore.exceptions import ClientError
from collections import defaultdict






def access_aws():
    """
    Accesses AWS S3 bucket
    :return: Returns all bucket object names, bucket name, host client
    """

    # Acces AWS S3 MATLAB file
    pubkey = mat_path['ACCESS_KEY']
    seckey = mat_path['SECRET_KEY']
    client = boto3.client(
        's3',
        aws_access_key_id = pubkey,
        aws_secret_access_key = seckey
    )
    s3 = boto3.resource(
        's3',
        aws_access_key_id = pubkey,
        aws_secret_access_key = seckey
    )
    bucket = s3.Bucket('teambrainiac')
    bucket_ = bucket.name
    obj_name = list(bucket.objects.all())

    return obj_name, bucket_, client





def create_paths():
    """
    Accesses AWS S3 to extract objects in bucket
    and create paths to access these items later
    when running and building datasets

    closes pickle file dictionary of bucket object paths
    """
    # Create a dictionary to store data values, subject IDs
    data_path_dictionary = defaultdict(list)

    # String vars
    substring_data = 'svm_subj_vecs.mat'
    substring_mask = 'masks.mat'
    substring_label = 'rt_labels.mat'
    sub_ID_regex = r"(\d{5}_\d{5})"  # extracts 10 digit ID separated in middle by underscore

    # Get objects in bucket
    obj_name, _, _ = access_aws()

    # Populate the dictionary
    for i in obj_name:
        if substring_data in i.key:
            data_path_dictionary['subject_data'].append(i.key)
            data_path_dictionary['subject_ID'].extend(re.findall(sub_ID_regex, i.key))
        if substring_mask in i.key:
            data_path_dictionary['mask_data'].append(i.key)
        if substring_label in i.key:
            data_path_dictionary['labels'].append(i.key)

    f = open("data/data_path_dictionary.pkl", "wb")
    pickle.dump(data_path_dictionary, f)
    f.close()






def data_to_nib(path):
    """
    Using Nibabel to open NiFTI files

    :param  nifti data path
    :return loads nifti data
    """

    filename = os.path.join(path)
    return nib.load(filename)






def load_mat(path):
    """
    uses scipy to convert mat file
    to be used in python

    :param  .mat data path
    :return  mat data

    """
    mat_file = scipy.io.loadmat(path)
    return mat_file






def open_pickle(file_path):
    """
    :param  opens pickle files
    :return data from pickle
    """
    f = open(file_path, "rb")
    data = pickle.load(f)
    f.close()

    return data






def access_load_data(obj, bool_mat):
    """
    :param obj        file path to data. For example can be pickle file containing a dictionary
                      or csv file path to load as a dataframe, e.g. file_names_dict['subject_data'][0]
                      or nifti file path to load as nifti object
                      or load a matlab file if bool_mat == TRUE
                      Stores the data in a temporary file before loading

    :param bool_mat   if true will run load_mat() and return .mat file

    :return           will open matlab data if bool_mat == True,
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
    :param data:         The data to upload
                         .npy, .csv or .nifit file
    :param object_name:  (String) S3 object name. If not specified then name of temp.name is used
    :param data_type:    (String) type of data file we are creating "pickle" "numpy" "csv" "nifiti"
    :return:             True if file was uploaded, else False
    """

    # Upload the file
    # Connect to AWS clientm return bucket name and client
    _, bucket_name, client = access_aws()

    try:

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            if data_type == "pickle":
                pickle.dump(data, temp, protocol = pickle.HIGHEST_PROTOCOL) # Protocols are finicky

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





def load_mask_indices(data_paths, mask_type, m_path_ind):
    """

    :param data_paths:  dictionary containing paths to data on AWS
    :param mask_type:   (String) type of mask
    :param m_path_ind:  (int) either 0 or 1, 0 for submasks, 1 for ROI masks
    :return:            Matrix or array of index values
    """
    mask_data_path = data_paths['mask_data'][m_path_ind]
    mask_type_dict = access_load_data(mask_data_path, True)
    np_array_mask = mask_type_dict[mask_type]
    print("mask shape:", np_array_mask.shape)
    np_compatible_mask = np.ma.make_mask(np_array_mask).reshape(79 * 95 * 79, order='F')
    indices_mask = np.where(
        np_compatible_mask == 1)  # gets the indices where the mask is 1, the brain region for x, y, z planes

    return indices_mask

    

    
