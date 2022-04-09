#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    access_data.py
# @Author:      staceyrivet
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

    :return:
    """

    # Acces AWS S3 MATLAB file
    pubkey = mat_path['ACCESS_KEY']
    seckey = mat_path['SECRET_KEY']
    client = boto3.client('s3', aws_access_key_id = pubkey, aws_secret_access_key = seckey)
    s3 = boto3.resource('s3', aws_access_key_id = pubkey, aws_secret_access_key = seckey)
    bucket = s3.Bucket('teambrainiac')
    bucket_ = bucket.name
    obj_name = list(bucket.objects.all())

    return obj_name, bucket_, client





def create_paths():
    """

    :return:
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
        # print(type(i.key))
        # print(i.key)
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
    _, bucket_name, client = access_aws()

    try:

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            if data_type == "pickle":
                pickle.dump(data, temp, protocol=pickle.HIGHEST_PROTOCOL)

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

    :param data_paths:
    :param mask_type:
    :param m_path_ind:
    :return:
    """
    mask_data_path = data_paths['mask_data'][m_path_ind]
    mask_type_dict = access_load_data(mask_data_path, True)
    np_array_mask = mask_type_dict[mask_type]
    print("mask shape:", np_array_mask.shape)
    indices_mask = np.where(
        np_array_mask == 1)  # gets the indices where the mask is 1, the brain region for x, y, z planes

    return indices_mask

    

    