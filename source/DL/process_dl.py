# File for preprocessing steps

import pickle
from path_config import mat_path
import boto3
import scipy
import pandas as pd
import nibabel as nib
import random

from access_data_dl import *
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




def load_subjects_by_id(data_path_dict, subject_ids, image_label_mask, image_labels, label_type='rt_labels', runs=[2, 3]):
  '''
    Function to load subject data. This deletes images with no labels and returns only the runs of interest for each subject.
    Params:
      data_path_dict  : dictionary containing information on how to access data
      subject_ids     : list of subjects to load by ids
      image_label_mask: a mask indicating whether a binary label exists for each image in a run
      image_labels    : binary labels indicating whether a subject was up or down regulating 
      label_type      : the type of label to return from the labels file in AWS 
      runs            : a list of runs to return from each subject
    returns: dictionary of users and their runs
  '''
  # Load subject Ids
  path_indicies = []
  for id in subject_ids:
    for i,path in enumerate(data_path_dict['subject_data']):
      if id in path.split('/'):
        path_indicies.append(i)
  
  subject_paths = []
  for index in path_indicies:
    subject_paths.append(data_path_dict['subject_data'][index])
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






def mask_normalize_runs_reshape_4d(subject_dict, mask, scaler='standard'):
    """
    subject_dict : subject data returned from load_subjects_chronologically or load_subjects_by_id function, keys are subject IDs)
    mask               : The mask meant to be applied to each image, typically a whole brain mask
    scaler             : Scaler from sklearn used to normalize the data, typically 'standard'
    
    returns            : dictionary with fully normalized subjects and labels
                         each subject's run is in 5d - n_images x n_color_layers(i.e. 1 because images are black and white) x length x width x height
    """

    runs_normalized_subjects = {}
    for i,sub_id in enumerate(subject_dict.keys()):
      temp_subject = {}
      for key in subject_dict[sub_id].keys():
        if key == 'image_labels':
          # Labels for the images
          temp_subject['image_labels'] = subject_dict[sub_id]['image_labels']
        
        else:
          # Subject Runs
          temp_run_unmasked = []
          temp_run = subject_dict[sub_id][key]

          if scaler == 'standard':
            temp_scaler = StandardScaler()
          else:
            print('Please import required scaler and update function')
            break
          
          temp_run_masked = temp_run[:,mask]
          temp_run_masked_norm =  temp_scaler.fit_transform(temp_run_masked)

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
          temp_run_unmasked_4d = []
          for image in temp_run_unmasked:
            image_3d = image.reshape((79,95,79), order='F')
            image_4d = np.array([image_3d])
            temp_run_unmasked_4d.append(image_4d)
          temp_run_unmasked_4d = np.array(temp_run_unmasked_4d)
          temp_subject[key] = temp_run_unmasked_4d

      runs_normalized_subjects[sub_id] = temp_subject
      subject_dict[sub_id] = None
      
      print('Completed Subject', str(i+1))

    return runs_normalized_subjects
  
  
  
 

def generate_train_val_test_dict(subject_id_partition, train_val_test_proportion=[0.7,0.8,1]):
  train_val_test_dict = {}
  shuffled_ids = subject_id_partition.copy()
  random.shuffle(shuffled_ids)
  train_index = int(train_val_test_proportion[0]*len(subject_id_partition))
  val_index = int(train_val_test_proportion[1]*len(subject_id_partition))
  
  train_val_test_dict['train'] = shuffled_ids[:train_index]
  train_val_test_dict['val'] = shuffled_ids[train_index:val_index]
  train_val_test_dict['test'] = shuffled_ids[val_index:]
  
  return train_val_test_dict
  
  

  
def train_test_aggregation_individual(single_subject, train_runs=[2], test_runs=[3,4]):
  """
    This function aggregates a single subjects runs into a training and test set
    split up by the desired runs, prepared for dataloader object
    single_subject     : A single subject's data from dictionary returned from mask_normalize_runs_reshape_3d
    train_runs         : List of Runs used in training set for a subject
    test_run           : List of Runs used in testing set for a subject
    
    returns            : train and test images and their labels ready for dataloader
  """
  # Training Data
  train = {}
  train_images = []
  train_labels = []

  for train_run in train_runs:
    run_key = 'run_'+str(train_run)
    for image in single_subject[run_key]:
      train_images.append(image)
    train_labels.extend(list(single_subject['image_labels']))

  train['images'] = torch.from_numpy(np.array(train_images).astype(float))
  train['labels'] = torch.from_numpy(np.array(train_labels).astype(float)).long()
  print('Train Runs Done')


  # Testing Data
  test = {}
  test_images = []
  test_labels = []

  for test_run in test_runs:
    run_key = 'run_'+str(test_run)
    for image in single_subject[run_key]:
      test_images.append(image)
    test_labels.extend(list(single_subject['image_labels']))

  test['images'] = torch.from_numpy(np.array(test_images).astype(float))
  test['labels'] = torch.from_numpy(np.array(test_labels).astype(float)).long()
  print('Test Runs Done')

  return train, test







def train_test_aggregation_group(subjects_dict, runs, train_val_test_ids):
  """
    This function aggregates a group of subjects' runs into a training and test set
    split up by the desired proportion
    subjects_dict       : A dictionary of subject images with ids as keys 
    runs                : List of Runs to use from subject dict
    train_val_test_ids : dictionary of subject ids for train, validation, and test 
    
    returns            : train and test images and their labels ready for dataloader or to save on AWS s3
  """
  
  partition = {}
  images = []
  labels = []
  for subject_id in train_val_test_ids:
    for run in runs:
      run_key = 'run_'+str(run)
      subject_images = subjects_dict[subject_id][run_key]
      for image in subject_images:
        images.append(image)
      labels.extend(list(subjects_dict[subject_id]['image_labels']))
  
  partition['images'] = torch.from_numpy(np.array(images).astype(float))
  partition['labels'] = torch.from_numpy(np.array(labels).astype(float)).long()
  
  return partition
