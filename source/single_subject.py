#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    single_subject.py
# @Author:      staceyrivet, benmerrill, marysoules
# @Time:        4/8/22 1:51 PM
# @IDE:         PyCharm


# Import libraries
import pickle
#sklearn packages needed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
#important utility functions for loading,masking,saving data
#from utils import *
from access_data import *
from process import *
#normal python packages we use
import numpy as np
#import nilearn clean for processing data
from nilearn.signal import clean

def get_data_dict(path='data/data_path_dictionary.pkl'):
  """
    Function to get data path dict
      params:
        path : str: path to data path dictionary set default to our data path dictionary
      returns: dictionary of data paths
  """
  data_path_dict = open_pickle(path)
  return data_path_dict

def get_subj_information(data_path_dict):
  """
    Function to get subject information.
      params:
        data_path_dict  : dictionary: containing paths to all data stored on AWS
      returns:  subject_ids(list of subjects to run),subj_paths(paths to subject raw data)
  """
  subject_ids = data_path_dict['subject_ID'] #subject_ids
  subj_paths = data_path_dict['subject_data'] #subject_paths
  return subject_ids,subj_paths

def get_labels(data_path_dict):
  """
    Function to get the labels for our data.
      params:
        data_path_dict  : dictionary: containing paths to all data stored on AWS
      returns: mask_labels_indices(timepoints we want masked out),binary_labels(labels forour two brain states)
               and label_type
  """
  
  label_data_path = data_path_dict['labels'][0] #get labels
  label_type = 'rt_labels' #tell the function what labels we want
  mask_labels_indices, binary_labels = labels_mask_binary(label_data_path, label_type) #grab indices and labels
  return mask_labels_indices, binary_labels,label_type

def get_mask_data(data_path_dict,mask_ind):
  """
    Function to return the mask of what brain voxels we want to include in analysis
    Params:
      data_path_dict  : dictionary: containing paths to data
      mask_ind: int: index of where the path to the masks are 0: full brain mask plus masks that subtract region
                1: Regions of interest(ROIs) mask out full brain except structure we care about
    returns: dictionary: contains mask data
    
  """
  mask_data_filepath = data_path_dict['mask_data'][mask_ind] #path to masked data     
  mask_type_dict = access_load_data(mask_data_filepath, True) #get the mask data dictionary
  
  return mask_type_dict

def make_mask(np_array_mask):
  """
    Function to create boolean mask to mask out voxels we don't want
    Params:
      mask_type: string: which mask to grab to get boolean array
    returns: boolean array of voxels to include
  """
  #np_array_mask = mask_data[mask_type] #get the mask array
  #create a 1-D array for the mask. Important to use Fourier Transformation as we are working in brain space!
  mask = np.ma.make_mask(np_array_mask).reshape(79*95*79,order='F')
  ind = np.where(mask==True)
  return mask,ind

def mask_subject_data(data,mask,mask_labels_indices):
  """
    Function to mask user data to mask out voxels we don't want
    Params:
      data: dictionary: subject data dictionary contain 4 runs of unmasked data
      mask: nd.array: 1-d array boolean values used to only include voxels we want.
      mask_labels_indices: indices of rows we want in to include in our model
    returns: dictionary: includes 4 runs of masked data
  """
  user_data_dict = {} #create empty dict
  arr = []
  for i in range(4):
      user_key = 'run_0' + str(i+1) + '_vec'
      array = data[user_key]
      array_masked = array[:, mask]
      array_masked = array_masked[mask_labels_indices]  
      arr.append(array_masked)
  user_data_dict['data'] = arr
  return user_data_dict


def scale_data_single_subj(sub_data,runs_list,norm='none'):
  """
    Function to scale data.
    Params:
      sub_data     : dictionary: contains subject run data, keys are run number for fmri data or labels)
      runs_list    : list: which runs are we normalizing on
      norm         : str: which type of normalization
    returns      : dictionary of nd.arrays, containing scaled runs of subject
                  
    """
  ##run standardization
  ##initialize empty dictionary
  normalized_runs = {}
  for run in runs_list:
    run_name = user_key = 'run_0' + str(run) 
    run_data = sub_data['data'][run-1]
    if norm=='none':
      normalized_runs[run_name] = clean(run_data,detrend=True,standardize=False,filter=False,standardize_confounds=False)
    else:
      normalized_runs[run_name] = clean(run_data,detrend=True,standardize=norm,filter=False,standardize_confounds=False)
  return normalized_runs

def get_accuracy_scores(clf,data,X_train,y_train,runs_test,y_labels):
  """
    Function to get accuracy scores for subject models.
    Params:
      clf: model: contains subject model
      data: contains subject test data to predict on
      X_train: nd_array: run data we trained on
      y_train: list: labels for x_train
      runs_test: list: runs to make predictions on
      y_labels: list: test labels (note in our paradigm all runs are the same so we only need one set)
    returns: subj_list, list of subject metrics, df_columns to map to dataframe
  """
  accuracy_list = [] #initialize accuracy score list
  df_columns = ['train_acc'] #initialize column list
  y_predicts = clf.predict(X_train) #get train acc
  accuracy_list.append(accuracy_score(y_train,y_predicts)) #append scores
  #iterate over runs and predict
  for run in runs_test:
    y_predicts = clf.predict(data[run]) #get predicted values
    df_columns.append(run + '_acc') #append run to columns
    accuracy_list.append(accuracy_score(y_labels,y_predicts)) #append scores
    df_columns.append(run+'_f1_score') #append run to columns
    accuracy_list.append(f1_score(y_labels,y_predicts)) #append scores
    
    
  return accuracy_list,df_columns

def get_predicts(clf,data,runs_test):
  """
    Function to get predictions for subject models.
    Params:
      clf: model: SVC model of subject
      data: dictionary: contains subject run information
      runs_test: list: runs to get test data on
    returns: dictionary of predictions we want to capture
  """
  predictions_dict = {} #initialize dictionary
  #iterate over runs and store data we want to save
  for runs in runs_test: 
    predictions_dict[runs] = {} #initalize empty dataframe
    predictions_dict[runs]['predicts'] = clf.predict(data[runs]) #get predicts
    predictions_dict[runs]['proba'] = clf.predict_proba(data[runs]) #get probability scores
    predictions_dict[runs]['decision_function'] = clf.decision_function(data[runs]) #get decision function scores

                                                  
  return predictions_dict

def run_single_subject_svm(data,runs_train,train_labels,svc_kernel='rbf',svc_gamma='scale',svc_c=1,do_cv=False,params={}):
  """
    Function to run cross-validation or single subject SVM
    Params:
      data: dictionary: contains all subject runn data
      runs_train: list: of runs we want to train on
      train_labels: list: list of labels for training data
    Optional Params:
      svc_kernel: str: string of kernel we want to use
      svc_gamma: str or float or int: of gamma we want to use
      svc_c: int or float: of C value we want to use
      do_cv: boolean: if true, run cross validation
      params: dictionary of params we want to use in cross validation
    returns: if do_cv, just the clf, if not cv, returns clf, X_train, y_train
  """ 
  
  X_train = [] #initialize X_train
  y_train = [] #initialize y_train
  
  if len(runs_train)>1: #if runs_train is more than one run
    #iterate over runs and concatenate to X_train and y_train
    for run in runs_train:
      X_train.append(data[run])
      y_train.append(train_labels)    
    X_train = np.concatenate(np.array(X_train))
    y_train = np.concatenate(np.array(y_train))
  else:
    X_train = data[runs_train[0]]
    y_train = train_labels
  #run cv if do_cv = True, else run individual model SVM
  if do_cv:
    svc = SVC()
    clf = GridSearchCV(svc, params)
    clf.fit(X_train,y_train)
    return clf
  else:
    clf = SVC(C=svc_c,kernel=svc_kernel,gamma=svc_gamma,probability=True)
    clf.fit(X_train,y_train)
  return clf,X_train,y_train

