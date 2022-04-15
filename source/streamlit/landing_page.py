#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    landing_page.py
# @Author:      staceyrivet
# @Time:        4/15/22 11:44 AM
# @IDE:         PyCharm

import streamlit as st



st.title("Team Brainiacs conquer fMRI")

PAGES = [
    'Brain images',
    'Chart Metrics',
    'Tables'
]



st.sidebar.title('Explore Data')


page = st.sidebar.radio('Navigation', PAGES)


if page == 'Brain images':
    st.sidebar.write("""
            ## About
            These images display the BOLD (Blood oxygen level dependent) voxel signal in the brain
            thresholded at a specific value. Areas in white/yellow indicate areas of the brain where
            blood volume increased and increases in oxygen exchange occur. This indicates that this part 
            of the brain is active when the image in the MRI is captured. Areas in the brain that are blue are
            when the blood volume and oxygen exchange decrease, indicating that these areas of the brain as becoming
            less active.
                
            """)


elif page == 'Chart Metrics':
    st.sidebar.write("""
            ## About
            The charts on this page are metrics captured when running our models on single subjects or when grouping subjects
            by age. Adolescent are ages 16-19 years old and Young Adults are older than 19 years old. 
            """)

else:
    st.sidebar.write("""
            ## About
            The tables on this page provide use with useful information about our decision choices for further
            preprocessing of our data. 
            """)



