# File for preprocessing steps

import pickle
from path_cofig import mat_path
import boto3
import scipy
import pandas as pd
import nibabel as nib

from utils_dl import *
import numpy as np
import tqdm


def get_mask(mask_type,data_path_dict,mask_ind):
  """
    Function to return the mask of what brain voxels we want to include in analysis
    Params:
      data_path_dict  : dictionary containing paths to all data stored on AWS
      mask_type: name of mask we want to use
      mask_ind: index of where the path to the masks are 0: full brain mask plus masks that subtract region
              1: Regions of interest(ROIs) mask out full brain except structure we care about
  """
  mask_data_filepath = data_path_dict['mask_data'][mask_ind] #path to masked data     
  mask_type_dict = access_load_data(mask_data_filepath, True) #get the mask data dictionary
  np_array_mask = mask_type_dict[mask_type] #get the mask array
  mask = np.ma.make_mask(np_array_mask).reshape(79*95*79,order='F') #create a 1-D array for the mask. Important to use Fourier Transformation as we are working in brain space!

  return mask



def labels_mask_binary(data_path_dict, label_type='rt_labels'):
    """
    """
    label_path = data_path_dict['labels'][0]
    label_data_dict = access_load_data(label_path, True)
    labels = label_data_dict[label_type].T[0]
    image_label_mask = np.array([bool(x!=9999) for x in labels])
    image_labels = labels[image_label_mask]

    return image_label_mask, image_labels




def load_subjects_chronologically(data_path_dict, n_subjects, image_label_mask, image_labels, label_type='rt_labels', runs=[2, 3]):
  '''
    Function to load subject data. This deletes images with no labels and returns only the runs of interest for each subject.
    Params:
      data_path_dict  : dictionary that has paths to data on AWS
      n_subjects      : the number of subjects
      image_label_mask: a mask indicating whether a binary label exists for each image in a run
      image_labels    : binary labels indicating whether a subject was up or down regulating 
      label_type      : the type of label to return from the labels file in AWS 
      runs            : a list of runs to return from each subject

    returns: dictionary of users and their runs
  '''
  # Load subject Ids
  subject_paths = data_path_dict['subject_data'][:n_subjects]
  subject_ids = data_path_dict['subject_ID'][:n_subjects]
  subjects = {}

  print('Subject ids loaded.\nAdding subjects to dictionary.')
  
  for path,id in tqdm.tqdm(zip(subject_paths, subject_ids)):
    subject_dict = {}
    data = access_load_data(path,True)
    subject_dict['image_labels'] = image_labels

    for run in runs:
      run_key = 'run_0' + str(run) + '_vec'
      run_masked = data[run_key][image_label_mask]
      subject_dict['run_'+str(run)] = run_masked
    
    subjects[id] = subject_dict
  
  return subjects
