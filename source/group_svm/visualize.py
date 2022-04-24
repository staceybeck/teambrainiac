#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    visualize.py
# @Author:      staceyrivet
# @Time:        3/30/22 10:09 PM


import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.signal import clean
import matplotlib.animation as animation






def five_sub_ani(data1, title, save_path, n_sub):
    """
    Creates an animated plot of histogram frames

    :param data1:       array of 2d data (time points, voxels)
    :param title:       (string) title for plot
    :param save_path:   (string) path to save
    :param n_sub:       (int) number of frames to make
    :return:            Animated plot
    """
    number_of_frames = n_sub
    data2 = np.array(data1)
    num_bins = 1000

    def update_hist(num, data2):
        plt.cla()
        plt.hist(data2[num],
                num_bins,
                facecolor='lightblue',
                log = True,
                range = (-.0000001, .0000001)
                )

    fig, axs = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    fig.suptitle(title)
    hist = axs.hist(data2[0],
                    num_bins,
                    facecolor='lightblue',
                    log = True,
                    range = (-.0000001, .0000001)
                    )

    ani = animation.FuncAnimation(fig,
                                  update_hist,
                                  number_of_frames,
                                  interval=500,
                                  fargs=(data2, )
                                  )

    with open(save_path, "w") as f:
        print(ani.to_html5_video(), file=f)

    return ani






def plot_dist_across_subjects(data, run, num_bins, sub_ids, norm, n_sub, data_type, detrend, plot_img):
    """
    plots distribution for n subjects for a given type of normalization

    :param data     : (dict: keys: 'string', values: nd.array) 52 subject data stored in a dictionary
    :param run      : (int) which run is being represented (out of 4) per subject
    :param num_bins : (int) how many bins to create for histogram
    :param sub_ids  : (string) represents the subject id
    :param norm     : (string) Options: 'Z-score', 'Percent Signal Change','Unnormalized'
    :param n_sub    : (int) how many subjects do we want to look at
    :param data_type: (string) type of group to specify (adolescent or young adult)
    :param detrend  : (bool) to detrend or not detrend, that is the question
    :param plot_img : (bool) to plot or not to plot the image, that is another question
    :return         : Plots # of subject subplots representing # of subjects for specified run and
                      normalization type.

    """
    small_set = []
    hist_sub_ids = sub_ids[:n_sub]

     # Choose time point 45
    for id_ in hist_sub_ids:
        # print(data[id_][run][44:45,:].shape)
        if norm == "Z-score":
            print(f"Running {norm}...")
            x = clean(data[id_][run],
                standardize='zscore',
                detrend=detrend,
                filter=False,
                standardize_confounds=False
                )
            small_set.append(np.array(np.mean(x, axis=0)))  # Average the timepoints #d[44:45,:])
        elif norm == "Percent Signal Change":
            print(f"Running {norm}...")
            x = clean(data[id_][run],
                standardize='psc',
                detrend=detrend,
                filter=False,
                standardize_confounds=False
                )
            small_set.append(np.array(np.mean(x, axis=0)))
        elif norm == 'Unnormalized':
            print("No Normalization...")
            x = clean(data[id_][run],
                standardize=False,
                detrend=detrend,
                filter=False,
                standardize_confounds=False
                )
            small_set.append(np.array(np.mean(x, axis=0)))

    if plot_img == False:
        return small_set
    else:
        # Create subplots
        fig, axs = plt.subplots(len(small_set[:n_sub]))
        fig.set_figheight(20)
        fig.set_figwidth(10)
        fig.suptitle(f'\n\n\n\n\nDetrended {norm} Voxel Histogram')

        for indx, arr in enumerate(small_set[:n_sub]):
            if norm == "Z-score":
                axs[indx].title.set_text(f"{data_type} Subject {indx + 1} Averaged Across Run {run + 1} Timepoints")
                axs[indx].hist(arr,
                       num_bins,
                       facecolor='lightblue',
                       log=True,
                       range=(-.0000001, .0000001)
                       )  # removed outliers
            elif norm == "Percent Signal Change":
                axs[indx].title.set_text(f"{data_type} Subject {indx + 1} Averaged Across Run {run + 1} Timepoints")
                axs[indx].hist(arr,
                               num_bins,
                               facecolor='lightblue',
                               log=True,
                               range=(-.0000001, .0000001)
                               )  # removed outliers
            elif norm == 'Unnormalized':
                axs[indx].title.set_text(f"{data_type} Subject {indx + 1} Averaged Across Run {run + 1} Timepoints")
                axs[indx].hist(arr,
                               num_bins,
                               facecolor='lightblue',
                               log=True
                               )  # , range = (-.0000001, .0000001))#, range = (-200, 200)) #, width = 1500)

        plt.savefig(f'/content/gdrive/MyDrive/sub_norm/five_{data_type}_run{run + 1}_vox_hist_{norm}.png', dpi=200)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])






def plot_dist_first_subject(data, num_bins, data_norm, detrend, data_type, plot_img):
    """
    Plots a single subject across all 4 runs

    :param data     : (dict: keys: 'string', values: nd.array) 52 subject data stored in a dictionary
    :param num_bins : (int) How many bins to create for histogram
    :param data_norm: (string) Options: 'Z-score', 'Percent Signal Change','Unnormalized'
    :param detrend  : (bool) to detrend or not detrend, that is the question
    :param data_type: (string) type of group to specify (adolescent or young adult)
    :param plot_img : (bool) to plot or not to plot the image, that is another question
    :return         : Plots 4 subplots representing the first subject for all 4 runs and type of represented data
                      as defined by data_norm.
    """

    subject = list(data.keys())[0]
    X_single_4run = data[subject]
    single_4run_norm = []

    if data_norm == "Z-score":
        print(f"Running {data_norm}...")
        for ind_ in range(len(X_single_4run)):
            x = clean(X_single_4run[ind_],
                     standardize='zscore',
                     detrend=detrend,
                     filter=False,
                     standardize_confounds=False
                     )
            single_4run_norm.append(np.array(np.mean(x, axis=0)))
    elif data_norm == "Percent Signal Change":
        print(f"Running {data_norm}...")
        for ind_ in range(len(X_single_4run)):
            x = clean(X_single_4run[ind_],
                    standardize='psc',
                    detrend=detrend,
                    filter=False,
                    standardize_confounds=False
                    )
            single_4run_norm.append(np.array(np.mean(x, axis=0)))
    elif data_norm == "Unnormalized":
        print("Unnormalized data...")
        for ind_ in range(len(X_single_4run)):
            x = clean(X_single_4run[ind_],
                    standardize=False,
                    detrend=detrend,
                    filter=False,
                    standardize_confounds=False
                    )
            single_4run_norm.append(np.array(np.mean(x, axis=0)))

    if plot_img == False:
        return single_4run_norm
    else:
        fig, axs = plt.subplots(len(single_4run_norm))
        fig.set_figheight(20)
        fig.set_figwidth(10)
        fig.suptitle(f'\n\n\n\nDetrended {data_norm} Voxel Histogram for {data_type} subject across {len(single_4run_norm)} runs')

        for indx, matrix in enumerate(single_4run_norm):
            if data_norm == "Z-score":
                axs[indx].title.set_text(f"{data_type} Subject Averaged Across Run {indx + 1} Timepoints")
                axs[indx].hist(matrix,
                               num_bins,
                               facecolor='lightblue',
                               log=True,
                               range=(-.0000001, .0000001)
                               )  # , range = (-200, 200)) # removed outliers
            elif data_norm == "Percent Signal Change":
                axs[indx].title.set_text(f"{data_type} Subject Averaged Across Run {indx + 1} Timepoints")
                axs[indx].hist(matrix,
                               num_bins,
                               facecolor='lightblue',
                               log=True,
                               range=(-.0000001, .0000001)
                               )  # , range = (-200, 200)) # removed outliers
            else:
                axs[indx].title.set_text(f"{data_type} Subject Averaged Across Run {indx + 1} Timepoints")
                axs[indx].hist(matrix,
                               num_bins,
                               facecolor='lightblue',
                               log=True
                               )  # range = (-.0000001, .0000001))

        plt.savefig(f'/content/gdrive/MyDrive/sub_norm/single_sub/{data_type}_vox_hist_{data_norm}.png', dpi=200)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])






def plot_brain_map(bmap3, bg_im, title):
    """

    :param bmap3:   3D Nifti Brain
    :param bg_im:   3D Nifti Background image (t1 weighted standard brain)
    :param title:   Title for plot
    :return:        displays the plot!
    """
    display = plotting.plot_stat_map(bmap3,
                                   bg_img = bg_im,
                                   colorbar=True,
                                   cmap='hot',
                                   display_mode='z',
                                   title=f"{title}"
                                   )
    return display






def plot_hist_decision(y_ng_dscore, y_ps_dscore, suptitle, savefile, if_save):
    """
    Plots the distribution of decision function scores for
    the positive and negative class

    :param y_ng_dscore:  array of floats - decision scores of the negative class
    :param y_ps_dscore:  array of floats - decision scores of the positive class
    :param suptitle:     (string) title for plot
    :param savefile:     (string) path to save to
    :param if_save:      (bool) to save or not to save, we must answer this question
    :return:
    """
    plt.style.use('seaborn-darkgrid')  # Style grid
    ax = plt.subplot()

    # plot the two histograms
    plt.hist(y_ps_dscore,
             bins=20,
             alpha=0.75,
             label='Up-regulating',
             # ls='dotted',
             # lw=.5,
             color='#446CCF',
             histtype='bar',
             # ec='olive'
             )
    plt.hist(y_ng_dscore,
             bins=20,
             alpha=0.75,
             label='Down-regulating',
             ls='dotted',
             # lw = .5,
             color='#f58518',
             histtype='bar',
             # ec='gold'
             )

    plt.xlim(-1.5, 1.5)
    plt.legend(loc="upper left")
    plt.xlabel('\nSVM decision function scores')
    plt.ylabel('Number of data points')
    plt.suptitle(suptitle)
    plt.title("\n Class Separation at Decision Boundaries")
    if if_save == True:
        print(f"Saving {savefile}...")
        plt.savefig(savefile,
                    dpi=200,
                    transparent=True
                    )
    plt.show()






def plot_decisions(ds, labels, time, title, file):
    """
    plots a line graph of one run at a time for
    decision scores

    :param ds:          array of floats of range of decision scores
    :param labels:      array of ints of labels
    :param time:        time period calculated for plot
    :param title:       string for title
    :param file:        string path for saving
    :return:            saves plot and returns
    """
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1, 1, figsize=(25, 5))
    ax.plot(ds,
            label='scaled SVM score\n',
            color="black"
            )
    ax.plot(labels,
            label='scaled true label values\n',
            color="#446ccf"
            )
    ax.set_xlabel('\ntime points [volumes]',
                  fontsize=10
                  )
    ax.axhline(0.0,
               ls="dotted",
               color='#b62020',
               label="> 0.0 Up-Regulating \n\n< 0.0 Down-Regulating"
               )
    ax.axvline(0,
               ls="dotted",
               color='gray'
               )
    ax.axvline(84,
               ls="dotted",
               color='gray'
               )
    ax.axvline(168,
               ls="dotted",
               color='gray'
               )

    ax.text(38,
            -.05,
            'Subject 4',
            color='black',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top'
            )
    ax.text(126,
            -.05,
            'Subject 5',
            color='black',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top'
            )
    ax.set_xticklabels([""])


    ax.set_title(f"{title}: Support Vector Decision Scores over {time} time points for test subjects 1 and 2")
    lgd = ax.legend(loc=(1.01, 0.5))
    plt.savefig(file,
                dpi=200,
                bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    plt.tight_layout()


