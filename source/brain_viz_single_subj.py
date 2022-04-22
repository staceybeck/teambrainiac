#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    single_subject.py
# @Author:      staceyrivet, benmerrill, marysoules
# @Time:        4/8/22 1:51 PM
# @IDE:         PyCharm

# Import libraries
#to load data
import pickle
#common packages we need
import numpy as np
import pandas as pd
#for brain imaging
import nibabel as nib
from nilearn import plotting, image
#for making graphs
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.image import threshold_img
#sklearn packages needed
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from single_subject import *
from access_data import *

def get_brain_templates(header_image='w3rtprun_01.nii',t1_template='single_subj_T1_resampled.nii',mni='MNI152.nii'):
  """
    Function to get brain templates used for brain visualizations
      Optional Params:
        header_image: str: name of image that contains the correct header information to transform our bmaps from arrays to nifti files
        t1_template: str: name of t1SPGR structural image to display under our bmaps
        mni        : str: name of brain template to display under our bmaps
      returns: Nifti Images: 
                affine_image: image to get header information (affine transformation)
                t1_image: subject t1SPGR that is in MNI space
                mni_image: MNI template image
  """
  #get image for header information
  affine_image = access_load_data(header_image,False)
  #get t1_image image for background
  #this image is a template of a warped brain to normalized space
  t1_image = access_load_data('single_subj_T1_resampled.nii',False)
  ## this is the mni image
  mni_image = access_load_data('MNI152.nii',False)
  return affine_image,t1_image,mni_image

def create_bmaps(data,indices_mask,image):
  """
    Function to compute beta maps using Lagrange values(alphas) from SVM.
      1. We want to create a zeroed vector to fill with the clf.dual_coef which correspond to the alphas
      2. We fill the vector with the alphas corresponding to the indices of the support vectors in clf.support_
      3. Reshape so we can find dot product
      4. Find dot prodoct between alphas and X_train data
      5. create a empty brain with dimensions 79*95*79
      6. reshape to a flattened vector
      7. fill the indices of the not masked voxels with the dot product
      8. reshape back to 3d brain
      9. create a nifti file of the brain for imaging
    Params:
      data: subject model and training/test data
      ind: indices of voxels to mask out
      image: template image to get header information for brain image
    Returns:
      bmap3: nifti file for imaging
      bmap2: numpy array to use for grabbing ROIs of interest on the whole brain model
  """
 
  clf = data['model'] # grab model
  X_train = data['X_train'] #grab training data
  y_train = data['y_train'] #grab labels

  alphas = np.zeros((84)) #create zero array
  alphas[clf.support_] = clf.dual_coef_ #fill indices with corresponding alphas
  alphas = alphas.reshape(1,-1) #reshape
  bmap = np.dot(alphas,X_train) #find dot product of X_train and alphas
  bmap2 = np.zeros((79,95,79)) #create empty 3-d brain
  bmap2 = bmap2.reshape(79*95*79) #flatten it to 1-d array
  bmap2[indices_mask] = bmap #fill indices of unmasked voxels w/the values created in bmap
  bmap2_3d = bmap2.reshape(79,95,79,order='F') #reshape back to a brain (3-d)
  bmap3 = nib.Nifti1Image(bmap2_3d,affine=image.affine,header=image.header) #save as nifti
  return bmap2,bmap3

def plot_map(bmap3_nifti,norm_type,subject_type,out_fname=None,threshold=.0001,t_image=None,display='z',cut_coord=(-35,-20,0,20,35,50,65,70)):
  """
    Function to plot beta map images from SVM models will either display image or save it to destination
    Params:
      bmap3: Nifti: image to plot
      norm_type: str: normalization strategy
      subject_type: str: young adult or adolescent
      outfname: str: if set to none will display image, other wise will save image to indicated outfname
      threshold: str or float: threshold for images, if voxel is less than threshold, it will not be shown
                               if want to see whole-voxel activation set to very low number as above
                               Nilearn plot_img documentation: https://nilearn.github.io/modules/generated/nilearn.plotting.plot_img.html
      t_image: nifti image to display in background, if none, displays none.
  """
  disp_image = None ##set image to nothing
  disp_image = bmap3_nifti ##set image to bmap3
  title_str = f'{norm_type} for {subject_type}' ##format string for title of image
  
  if out_fname!=None: #if outfname specified
    output_file_name = f'{out_fname}{norm_type}_{subject_type}.png'
    disply = plotting.plot_img(disp_image, bg_img = t_image, display_mode=display, cut_coords=cut_coord, threshold=threshold,
                          output_file=output_file_name, colorbar=True,cmap='cold_white_hot',black_bg=False)
  else:
    display = plotting.plot_img(disp_image, bg_img = t_image, display_mode=display, cut_coords=cut_coord,threshold=threshold,
                           colorbar=True,cmap='cold_white_hot',black_bg=False)
    display.title(title_str, x=0.01, y=0.99, size=15, color='w', bgcolor='black')
    

def plot_decision_scores(scores,labels,title_str,subject_type,run,outfname=None):
  """
    Function to plot decision function scores.
    Params:
      scores: list of decision function scores
      labels: labels for data 0 = decrease, 1 = increase
      title_str: how do we want the title to display
      subject_type: is this a young adult or adolescent (added to title of visualization)
      run: which run are scoring
      outfname: if None, display, if str, save data as .png image and display
  """

  fig, ax = plt.subplots(1,1,figsize=(15, 2))
  plt.style.use('seaborn-darkgrid')
  ax.plot(labels, lw=3, c='#446CCF',label='True Labels\n')
  ax.plot(scores,c='k',lw=0.5,linestyle='-',label='Decision Function Scores\n')
  ax.axhline(y=0,c='r',linestyle='--',label='Decision Function Cutoff\n')
  ax.set_xlabel('\ntime points [volumes]',
                  fontsize=10
                  )
  ax.set_title(f'{subject_type} Decision Function Scores for {title_str} on {run}')
  lgd = ax.legend(loc=(1.01, 0.5))
  if outfname!=None:
    file = f'{outfname}{subject_type}_descf_{title_str}.png'
    print(file)
    plt.savefig(f'{outfname}{subject_type}_descf_{title_str}_{run}.png',dpi=200,bbox_extra_artists=(lgd,),
                bbox_inches='tight')
  else:
    plt.show()
    
#   x = [0]
#   y = [0]
#   decf_labels = np.append(labels,0)
#   fig, ax = plt.subplots(1,1,figsize=(15, 2))
#   plt.style.use('seaborn-darkgrid')
#   for tp in range(1,85): 
#       if decf_labels[tp-1] == decf_labels[tp]:
#         x.append(tp)
#         y.append(decf_labels[tp])
#       else:
#         y.append(0)
#         x.append(tp)
#         if y[1] == 1:
#           color = '#446CCF'
#         else: 
#           color='#f58518'      
#         ax.plot(x,y,c=color,lw=3)
#         ax.axhline(y=0,c='r',linestyle='--')
#         y=[0]
#         x=[tp]
#         ax.plot(scores,c='k',lw=0.2,linestyle='-')
#         #plt.legend(['Increase','Decision Function Cutoff','Decision Scores','Decrease'],bbox_to_anchor=(1,1.04), loc="upper left")
#         ax.set_xlabel('time [volumes]', fontsize=10)
#         ax.tick_params(labelsize=12)
#         ax.set_title(f'{subject_type} Decision Function Scores for {title_str} on {run}')
#         ax.legend()
#   if outfname!=None:
#     plt.savefig(f'outfname{subject_type}_descf_{title_str}.png',dpi=200)
#   plt.show()
  
