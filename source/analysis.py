#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    analysis.py
# @Author:      staceyrivet
# @Time:        4/6/22 7:32 PM
# @IDE:         PyCharm








from utils import *
from process import *
from cross_validation import *
import numpy as np
import nibabel as nib
from nilearn.image import threshold_img





def create_bmaps(clf, X, indices_mask, image):

    """


    :param clf:
    :param X:
    :param indices_mask:
    :param image:
    :return:
    """

    # Create alpha matrix and map weights to support vector indices
    alphas = np.zeros((len(X)))
    alphas[clf.support_] = clf.dual_coef_ #Load the weights corresponding to where support vectors are

    alphas = alphas.reshape(1,-1)
    bmap = np.dot(alphas, X)
    print("Shape of beta map: ", bmap.shape)

    # Grab the areas not masked out of the brain to recreate the brain using highlighted areas
    bmap2 = np.zeros((79,95,79))
    bmap2[indices_mask] = bmap

    bmap3 = nib.Nifti1Image(bmap2, affine = image.affine, header = image.header)

    return bmap3






def get_threshold_image(bmap3, score_percentile, image_intensity):
    """

    :param bmap3:
    :param score_percentile:
    :param image_intensity:
    :return:
    """
    # Two types of strategies can be used from this threshold function
    # Type 1: strategy used will be based on score percentile
    threshold_percentile_img = threshold_img(bmap3, threshold = score_percentile, copy = False)


    # Type 2: threshold strategy used will be based on image intensity
    # Here, threshold value should be within the limits i.e. less than max value.
    threshold_value_img = threshold_img(bmap3, threshold = image_intensity, copy = False)
    return threshold_percentile_img, threshold_value_img







