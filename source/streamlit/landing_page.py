#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    landing_page.py
# @Author:      staceyrivet
# @Time:        4/15/22 11:44 AM
# @IDE:         PyCharm

import streamlit as st
import streamlit.components.v1 as components



PAGES = [
    'Brain Images',
    'Chart Metrics',
    'Tables'
]

st.sidebar.title('Explore Data')
st.session_state.page_select = st.sidebar.radio('Pages', PAGES)

if st.session_state.page_select == 'Brain Images':
    st.title("Brain States")
    st.sidebar.write("""
            ## Brain Images
            These images display the BOLD (Blood oxygen level dependent) voxel signal in the brain
            thresholded at a specific value. Areas in white/yellow indicate areas of the brain where
            blood volume is increased and where increases in oxygen exchange occur. This indicates that this part 
            of the brain is active when the image in the MR machine is captured. Areas in the brain that are blue are
            when the blood volume and oxygen exchange decrease, indicating that these areas of the brain as becoming
            less active.
                
            """)
    display = ('Young Adult Whole Brain',
               'Young Adult without mPFC',
               'Young Adult Prefrontal Cortex',
               'Adolescent Whole Brain',
               'Adolescent Nucleus Accumbens',
               'Adolescent Prefrontal Cortex',
               )

    value = 0
    options = list(range(len(display)))

    def get_html(value):
        #print(value)
        if value == 0:
            HtmlFile = open("YA_YA_mask_1.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

        if value == 1:
            HtmlFile = open("YA_YA_masksubmPFC_1.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

    value = st.selectbox("Choose the type of brain activations to view:", options, format_func=lambda x: display[x])
    get_html(value)

if st.session_state.page_select == 'Chart Metrics':
    st.title("Chart Metrics")
    st.sidebar.write("""
            ## Chart Metrics
            The charts on this page are metrics captured when running our models on single subjects or when grouping subjects
            by age. Adolescent are ages 16-19 years old and Young Adults are older than 19 years old. 
            """)
    display = ('Young Adult Z-score Normalization',
               'Young Adult Percent Signal Change',
               'Young Adult Unnormalized',
               'Adolescent Z-score Normalization',
               'Adolescent Percent Signal Change',
               )

    value = 0
    options = list(range(len(display)))

    def get_html(value):
        # print(value)
        if value == 0:
            HtmlFile = open("YA_dtrnd_ZSCORE_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 1:
            HtmlFile = open("YA_dtrnd_psc_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 2:
            HtmlFile = open("YA_dtrnd_Unorm_mvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 3:
            HtmlFile = open("ADdtrnd_ZSCORE_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 4:
            HtmlFile = open("ADdtrndpscnormvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)


    value = st.selectbox("Choose the type of brain activations to view:", options, format_func=lambda x: display[x])
    get_html(value)


if st.session_state.page_select == 'Tables':
    st.title("Tables")
    st.sidebar.write("""
            ## Tables
            The tables on this page provide use with useful information about our decision choices for further
            preprocessing of our data. 
            """)



