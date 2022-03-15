"""
This file loads NiFTI data
using NiBabel Library


"""

import nibabel as nib
import os
import scipy.io
#import matplotlib.pyplot as plt
#from ipywidgets import interact
import pickle
from path_config import mat_path
import boto3
import tempfile
import numpy as np
from collections import defaultdict
import tqdm





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






def access_load_data(dict_file):
    """
    :param dict_file: dictionary loaded from pickle containing storage data paths
    e.g. file_names_dict['subject_data'][0]

    :return: matlab data
    """


    # Acces AWS S3 MATLAB file

    # Connect to AWS client
    pubkey = mat_path['ACCESS_KEY']
    seckey = mat_path['SECRET_KEY']
    client = boto3.client('s3', aws_access_key_id=pubkey, aws_secret_access_key=seckey)
    s3 = boto3.resource('s3', aws_access_key_id = pubkey, aws_secret_access_key = seckey)

    # Create a temporary file
    temp = tempfile.NamedTemporaryFile()

    # Grab bucket name
    bucket = s3.Bucket('teambrainiac')
    bucket_ = bucket.name  # 'teambrainiac'

    # Define object that we want to get in bucket
    obj = dict_file

    # Define
    client.download_file(bucket_, obj, temp.name)
    data = load_mat(temp.name)
    temp.close()

    return data





def save_data(data, local_file_path):
    """

    :param data: Will contain the data as .mat
    :param local_file_path: specified local path to save data

    Saves a numpy file of the data locally

    """
    with open(local_file_path, 'wb') as f:
        np.save(f, data)

    f.close()

    
    
    
        

def create_mask(mask_data_filepath, mask_type = 'mask'):
    """
    mask_data_filepath: Takes in filepaths 
    mask_type: and mask type 
    returns: a numpy-compatible mask
    
    """
    mask_type_dict = access_load_data(mask_data_filepath)
    np_array_mask = mask_type_dict[mask_type]
    np_compatible_mask = np.ma.make_mask(np_array_mask).reshape(79*95*79)
    
    return np_compatible_mask






def labels_mask_binary(label_data_path, label_type = 'rt_label'):
    """
    
    """
    # Get Mask data
    label_data_dict = access_load_data(label_data_path)
    labels = label_data_dict[label_type]
    mask_labels_indices = np.where(labels != 9999)
    mask_labels_indices = mask_labels_indices[0]
    
    # Get Binary Labels
    label_mask = np.all(labels != 9999, axis=1)
    binary_labels = labels.reshape(-1)[label_mask]
    binary_labels
    
    return mask_labels_indices, binary_labels






def masking_data(subject, mask, mask_labels, binary_labels):
    """
    
    
    """
    
    arr = []
    label_arr = []
    for i in tqdm.tqdm(range(4)):
        user_key = 'run_0' + str(i+1) + '_vec'
        array = subject[user_key]
        array = array[:, mask] 
        arr.append(array[mask_labels])
        label_arr.append(binary_labels)
    
    return arr, label_arr






def masked_data_n_labels(mask_type, label_type, path_dict):
    """
    mask_type: String for type of mask we want
    label_type: String for which labels we want
    path_dict: dictionary that contains all the paths for data in AWS
    Returns: a dictionary of subject masked data and the cleaned labels
    """
    # Define variable to return
    user_data_dict = defaultdict(list)
    
    # Mask path from S3
    mask_data_filepath = path_dict['mask_data'][0]
    mask_type = mask_type #'mask'
    mask = create_mask(mask_data_filepath, mask_type)
    
    # Label path from S3
    label_data_path = path_dict['labels'][0]
    label_type = label_type #"rt_labels"
    
    # Returns two values, mask_labels_indices and binary_labels
    mask_labels_indices, binary_labels = labels_mask_binary(label_data_path, label_type)
    
    # Loop through all subjects
    for ind, val in tqdm.tqdm(enumerate(path_dict['subject_ID'])):
        sub_id = val
        sub_path = path_dict['subject_data'][ind]
        subject = access_load_data(sub_path)
        user_data_dict[sub_id], bi_lb = masking_data(subject, mask, mask_labels_indices, binary_labels)
        user_data_dict[f"{sub_id}_{label_type}"] = bi_lb

    
    return user_data_dict







               





