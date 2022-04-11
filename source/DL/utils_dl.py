# General utility functions

import pickle
from path_config import mat_path
import boto3
import scipy
import pandas as pd
import nibabel as nib
import tempfile


def open_pickle(file_path):
    """
    :param file path for dictionary
    :return dictionary:
    """
    f = open(file_path, "rb")
    dictionary = pickle.load(f)
    f.close()

    return dictionary
  
    
    
    
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
