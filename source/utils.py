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
from collections import defaultdict
import tqdm
#from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean
from botocore.exceptions import ClientError
from utils import *
import numpy as np
import pandas as pd




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






def access_load_data(data_file, bool_mat):
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
    obj = data_file

    # Define
    client.download_file(bucket_, obj, temp.name)
    
    if bool_mat == True:
        data = load_mat(temp.name)
    else:
        if '.pkl' in data_file:
            data = open_pickle(temp.name)
        elif '.csv' in data_file:
            data = pd.read_csv(temp.name)
        elif '.nii' in data_file:
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
    mask_type_dict = access_load_data(mask_data_filepath, True)
    np_array_mask = mask_type_dict[mask_type]
    np_compatible_mask = np.ma.make_mask(np_array_mask).reshape(79*95*79)
    
    return np_compatible_mask






def labels_mask_binary(label_data_path, label_type = 'rt_labels'):
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
        user_key = 'run_0' + str(i+1) + '_vec'
        array = subject[user_key]
        array_masked = array[:, mask]
        array_masked = array_masked[mask_labels]
        
        # Percent Signal Change normalization
        if do_norm:
            array_masked = clean(array_masked,standardize='psc')
        
        arr.append(array_masked)
        label_arr.append(binary_labels)
    
    return arr, label_arr






def masked_data_n_labels(mask_type, label_type, path_dict, do_norm):
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
        subject = access_load_data(sub_path, True)
        user_data_dict[sub_id], bi_lb = masking_data(subject, mask, mask_labels_indices, binary_labels, do_norm)
        user_data_dict[f"{sub_id}_{label_type}"] = bi_lb

    
    return user_data_dict







               





