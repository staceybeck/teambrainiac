"""
This file loads NiFTI data
using NiBabel Library
"""

import nibabel as nib
import os
import scipy.io





def load_data(path):

    filename = os.path.join(path)
    return nib.load(filename)






def convert_mat(path):
    mat_file = scipy.io.loadmat(path)
    return mat_file
