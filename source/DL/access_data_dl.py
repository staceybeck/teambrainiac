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
                pickle.dump(data, temp, protocol = pickle.HIGHEST_PROTOCOL)

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
