#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    visualize.py
# @Author:      staceyrivet
# @Time:        3/30/22 10:09 PM





import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from nilearn import plotting






def plot_dist_across_subjects(data, run, num_bins, sub_ids, data_norm, n_sub):
  """

  :param data     : (dict: keys: 'string', values: nd.array) 52 subject data stored in a dictionary
  :param run      : (int) which run is being represented (out of 4) per subject
  :param num_bins : (int) how many bins to create for histogram
  :param sub_ids  : (string) represents the subject id
  :param data_norm: (string) how do we want to represent our data. Options: 'PSC_ZNORM', 'Z-Normalized',
                    'Unnormalized', 'Percent Signal Change'
  :param n_sub    : (int) how many subjects do we want to look at
  :return         : Plots n_sub subplots representing n_sub subjects for specified run and type of represented data
                    as defined by data_norm.
  """
  small_set = []
  hist_sub_ids = sub_ids[:n_sub]

  # Choose time point 45
  for id_ in hist_sub_ids:
    #print(data[id_][run][44:45,:].shape)
    if (data_norm == "PSC_ZNORM") or (data_norm == "Z-Normalized"):
      print(f"Running {data_norm}...")
      scaler = StandardScaler()
      d = scaler.fit_transform(data[id_][run])
      small_set.append(d[44:45,:])
    else:
      print("not running PSC_ZNORM or ZNROM ALONE...")
      small_set.append(data[id_][run][44:45,:])


  small_set = np.concatenate(np.array(small_set))
  print("sample 5 subject set size: ", small_set.shape)

  # Create subplots
  fig, axs = plt.subplots(len(small_set[:n_sub]))
  fig.set_figheight(12)
  fig.set_figwidth(10)
  fig.suptitle(f'{data_norm} Voxel Histogram for {n_sub} subjects for run {run + 1} at time point 45')


  for indx, arr in enumerate(small_set[:n_sub]):

    if data_norm == "PSC_ZNORM":
      axs[indx].hist(arr, num_bins, facecolor='blue', log = True, range = (-50, 50)) # removed outliers
    elif data_norm == "Percent Signal Change":
      axs[indx].hist(arr, num_bins, facecolor='blue', log = True, range = (-200, 200)) # removed outliers
    else:
      axs[indx].hist(arr, num_bins, facecolor='blue', log = True) #, width = 1500)


  plt.savefig(f'data/five_sub_run{run + 1}_vox_hist_{data_norm}.png')
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])





def plot_dist_first_subject(data, num_bins, data_norm):
  """

  :param data     : (dict: keys: 'string', values: nd.array) 52 subject data stored in a dictionary
  :param num_bins : (int) How many bins to create for histogram
  :param data_norm: (string) How do we want to represent our data. Options: 'PSC_ZNORM', 'Z-Normalized',
                      'Unnormalized', 'Percent Signal Change'
  :return         : Plots 4 subplots representing the first subject for all 4 runs and type of represented data
                      as defined by data_norm.
  """

  subject = list(data.keys())[0]
  X_single_4run = data[subject]
  single_4run_norm = []

  if (data_norm == "PSC_ZNORM") or (data_norm == "Z-Normalized"):
    print(f"Running {data_norm}...")
    scaler = StandardScaler().fit(X_single_4run[0])
    for ind_ in range(len(X_single_4run)):
      d = scaler.transform(X_single_4run[ind_])
      single_4run_norm.append(d)
  else:
    print("Not running PSC ZNORM...")
    single_4run_norm = data[subject]


  fig, axs = plt.subplots(len(single_4run_norm))
  fig.set_figheight(12)
  fig.set_figwidth(10)
  fig.suptitle(f'{data_norm} Voxel Histogram for subject: {subject} across {len(single_4run_norm)} runs')

  for indx, matrix in enumerate(single_4run_norm):
    if (data_norm == "PSC_ZNORM") or (data_norm == "Percent Signal Change"):
      axs[indx].hist(matrix[44:45,:][0], num_bins, facecolor='blue', log = True, range = (-200, 200)) # removed outliers
    else:
      axs[indx].hist(matrix[44:45,:][0], num_bins, facecolor='blue', log = True)

  plt.savefig(f'data/single_sub{subject}_vox_hist_{data_norm}.png')
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])







def plot_alphas(alphas,alpha_labels, time, title):
  """

  :param alphas:
  :param alpha_labels:
  :param time:
  :param title:
  :return:
  """
  fig, ax = plt.subplots(1,1,figsize=(15, 5))
  ax.plot(alphas, label='scaled voxel tc')
  ax.plot(alpha_labels,  label='predictor tc')
  ax.set_xlabel('time [volumes]', fontsize=20)
  ax.tick_params(labelsize=12)
  ax.set_title(f"Alpha signal for {title} {time} time points")
  ax.legend(loc = 'upper right')
  plt.show()






def plot_brain_map(bmap3, bg_im, title):
  """

  :param bmap3:
  :param bg_im:
  :param title:
  :return:
  """
  display = plotting.plot_stat_map(bmap3, bg_img = bg_im,
                                   colorbar=True, cmap='hot', display_mode='z',
                                   title=f"{title}")
  return display