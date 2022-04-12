# File for preprocessing steps

import pickle
from path_config import mat_path
import boto3
import scipy
import pandas as pd
import nibabel as nib

from utils_dl import *
import numpy as np
import tqdm

from sklearn.preprocessing import StandardScaler



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






def mask_normalize_runs_reshape_3d(chron_subject_dict, mask, scaler):
    """
    chron_subject_dict : (subject data returned from load_subjects_chronologically function, keys are subject IDs)
    mask               : The mask meant to be applied to each image, typically a whole brain mask
    scaler             : Scaler from sklearn used to normalize the data

    returns            : X and y data normalized based on across all runs for a subject or per subject for a single run
    """

    runs_normalized_subjects = {}
    for i,sub_id in enumerate(chron_subject_dict.keys()):
      temp_subject = {}
      for key in chron_subject_dict[sub_id].keys():
        if key == 'image_labels':
          # Labels for the images
          temp_subject['image_labels'] = chron_subject_dict[sub_id]['image_labels']
        
        else:
          # Subject Runs
          temp_run_unmasked = []
          temp_run = chron_subject_dict[sub_id][key]

          if scaler == 'standard':
            temp_scaler = StandardScaler()
          else:
            print('Please import required scaler and update function')
            break
          
          temp_run_masked = temp_run[:,mask]
          temp_run_masked_norm =  temp_scaler.fit_transform(subject_brain)

          temp_run_masked_norm_index = 0
          for t_or_f in mask:
            if t_or_f == True:
              temp_run_unmasked.append(temp_run_masked_norm[:,temp_run_masked_norm_index])
              temp_run_masked_norm_index += 1
            elif t_or_f == False:
              temp_run_unmasked.append(np.zeros(len(temp_run_masked_norm[:,0])))
            else:
              print('Error')
              break
        
          temp_run_unmasked = np.array(temp_run_unmasked).T
          temp_run_unmasked_3d = []
          for image in temp_run_unmasked:
            temp_run_unmasked_3d.append(image.reshape(79 * 95 * 79, order='F'))
          temp_run_unmasked_3d = np.array(temp_run_unmasked_3d)
          temp_subject[key] = temp_run_unmasked_3d

      runs_normalized_subjects[sub_id] = temp_subject
      
      print('Completed Subject', i)

    return runs_normalized_subjects
