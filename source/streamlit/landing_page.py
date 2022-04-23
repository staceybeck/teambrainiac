#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Project:     teambrainiac
# @Filename:    landing_page.py
# @Author:      staceyrivet
# @Time:        4/15/22 11:44 AM
# @IDE:         PyCharm

import streamlit as st
import streamlit.components.v1 as components
#from access_data import *
#import plotly.graph_objects as go
#import pandas as pd
#import numpy as np




PAGES = [
    'Brain Images',
    'Exploration',
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
    display = ('Young Adult and Adolescent Whole Brain',
               'Young Adult and Adolescent Medial Prefrontal Cortex',
               'Young Adult and Adolescent Nucleus Accumbens',
               )

    value = 0
    options = list(range(len(display)))

    def get_html(value):
        #print(value)
        if value == 0:
            st.write("Signals thresholded at 95%")
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_detrend_mask_1.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            HtmlFile = open("/app/teambrainiac/source/streamlit/AD_detrend_mask_1.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            st.write("This is an interactive brain map. We trained a Support Vector Machine on Young Adult Brains (aged 19 - 21) "
                     "and Adolescent Brains (aged 16 - 19) to predict states of up-regulation and down-regulation - meaning "
                     "an uptake in blood-oxygen volume or decrease in volume during an impulse-task. Areas that are yellow/red"
                     " are active brain areas during this task and areas that are dark-blue/light-blue are areas where blood and "
                     "oxygen levels are decreasing."
                     "\n"
                     "In our research, we have found that the Young Adult models predicted better than the Adolescent models, "
                     "which is evident in the two visuals. You can see clear and distinct areas of increased voxel significance "
                     "in the Young Adult model output with dark red and blue shades compared to the Adolescent model output, where "
                     "the significant signals are scattered and are light blue and yellow. ")

        if value == 1:

            st.write("Signals thresholded at 95%")
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_detrend_mPFC_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            HtmlFile = open("/app/teambrainiac/source/streamlit/AD_detrend_mPFC_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)



            st.write("These are interactive brain maps showing the Medial Prefrontal Cortex (mPFC) - an area of the brain researchers "
                     "know to be involved in the impulse-reward system. Each interactive visual is set to the same cut coordinates. We removed"
                     " the grid cross since these areas of the brain are small to visualize. "
                     "\n"
                     "In our Young Adult model, we can see there are large areas of the mPFC that are down-regulating, "
                     "meaning an decrease in activation during the impulse-reward task that the subjects are performing while "
                     " in the MR machine. "
                     "In the Adolescent model, we are seeing more yellow, which implies increased up-regulation in this region.")

        if value == 2:
            st.write("Signals thresholded at 95%")

            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_detrend_nacc_aal_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            HtmlFile = open("/app/teambrainiac/source/streamlit/AD_detrend_nacc_aal_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)
            st.write(
                "These are interactive brain maps showing the Nucleus Accumbens (NAcc)- an area of the brain researchers "
                "know to be involved in the impulse-reward system. Each interactive visual is set to the same cut coordinates. We removed"
                     " the grid cross since these areas of the brain are small to visualize."
                ""
                "\n"
                " In our Young Adult model, we can see there are large areas of red voxels in the NAcc, which imply these "
                " subjects are up-regulating; an increase in activation during the impulse-reward task that the subjects "
                " are performing in while in the MR machine."
                )

    value = st.selectbox("Interactive Brain Visualization. Choose the type of brain activation to view and use your cursor to move the grid cross:", options, format_func=lambda x: display[x])
    get_html(value)

if st.session_state.page_select == "Chart Metrics":
    st.title("Chart Metrics")
    st.sidebar.write("""
            ## Chart Metrics
            The charts on this page are metrics captured when running our models on single subjects or when grouping subjects
            by age. Adolescents range from 14 - 16 years old and Young Adults range from 25 - 27 years old. 
    """)


    st.write ("Young Adult Whole Brain Mask Model Decision Scores")
    st.image('/app/teambrainiac/source/streamlit/YA_detrend_mask_dfunpred_1.png',
             caption=None,
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")


if st.session_state.page_select == 'Exploration':
    st.title("Exploration")
    st.sidebar.write("""
            ## Exploration 
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
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_dtrnd_ZSCORE_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 1:
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_dtrnd_psc_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 2:
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_dtrnd_Unorm_mvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 3:
            HtmlFile = open("/app/teambrainiac/source/streamlit/ADdtrnd_ZSCORE_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=500)

        if value == 4:
            HtmlFile = open("/app/teambrainiac/source/streamlit/ADdtrndpscnormvid.html", 'r',
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
    st.image("/app/teambrainiac/source/streamlit/newplot.png",
             caption="Young Adult Model Scores",
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")
