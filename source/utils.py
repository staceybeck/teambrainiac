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









               





