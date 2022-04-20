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
import plotly.graph_objects as go
import pandas as pd
import numpy as np




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
            st.write("Young Adult")
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_detrend_mask_1.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            st.write("Adolescent")
            HtmlFile = open("/app/teambrainiac/source/streamlit/AD_detrend_mask_1.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            st.write("This is an interactive brain map. We trained a Support Vector Machine on Young Adult Brains"
                     "to predict states of up-regulation and down-regulation - essentially an uptake in blood-oxygen "
                     "volume or decrease in volume during an impulse-task. Areas that are white/yellow are active brain"
                     "areas during this task and areas that are black/blue are areas where blood and oxygen levels are decreasing.")

        if value == 1:

            st.write("Young Adult")
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_detrend_mPFC_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            st.write("Adolescent")
            HtmlFile = open("/app/teambrainiac/source/streamlit/AD_detrend_mPFC_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)



            st.write("This is an interactive brain map showing the Medial Prefrontal Cortex (mPFC) - an area of the brain researchers "
                     "know to be involved in the impulse-reward system. In our Young Adult model, we can see there are large areas"
                     "of the mPFC that are down-regulating, meaning an decrease in activation during the impulse-reward task that"
                     "the subjects are performing in while in the MR machine.")

        if value == 2:
            st.write("Young Adult")
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_detrend_nacc_aal_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)

            st.write("Adolescent")
            HtmlFile = open("/app/teambrainiac/source/streamlit/AD_detrend_nacc_aal_nocross.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=250)
            st.write(
                "This is an interactive brain map showing the Nucleus Accumbens (NAcc)- an area of the brain researchers "
                "know to be involved in the impulse-reward system. In our Young Adult model, we can see there are large areas"
                "of the NAcc that are up-regulating, meaning there is an increase in activation during the impulse-reward "
                "task that the subjects are performing in while in the MR machine.")

    value = st.selectbox("Interactive Brain Visualization. Choose the type of brain activation to view and use your cursor to move the grid cross:", options, format_func=lambda x: display[x])
    get_html(value)

if st.session_state.page_select == "Chart Metrics":
    st.title("Chart Metrics")
    st.sidebar.write("""
            ## Chart Metrics
            The charts on this page are metrics captured when running our models on single subjects or when grouping subjects
            by age. Adolescent are ages 16-19 years old and Young Adults are older than 19 years old. 
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
    # Load model
    # open path dictionary file to get subject ids
    """
    dict_path = "data/data_path_dictionary.pkl"
    data_path_dict = open_pickle(dict_path)

    mask_type = 'mask'  # 'mask', 'masksubACC', 'masksubAI', 'masksubNAcc', 'masksubmPFC'
    data_type = "YA_detrend"
    runs_train = [1, 2]
    runs_id = [i + 1 for i in runs_train]
    m_path_ind = 0  # get sub_mask data in mask_data key
    metric_data = access_load_data(f'metrics/group_svm/{mask_type}/{data_type}_{runs_id}_{mask_type}_metrics.pkl',
                                   False)
    # Create a DataFrame from metrics of Whole Brain
    decision_scores = metric_data['test_dfunc'][0]
    y = metric_data['y_t'][0]

    time = np.arange(0, len(decision_scores), 1)
    d = {}
    d['time'] = time
    d['score'] = decision_scores
    d['true'] = y
    df = pd.DataFrame(d)

    y_axis_frame = list(df['score'].values[:300])
    y_axis = list(df['score'].values[:300])
    y_ax = list(df['score'].values)
    x_axis_framestart = np.arange(0, 300, 1)
    x_axis_frame = np.arange(0, 300, 1)
    framess = []
    for frame in range(300, 673):
        x_axis_frame = np.append(x_axis_frame, frame)
        x_axis_frame = x_axis_frame[1:]

        y_axis_frame = np.append(y_axis_frame, y_ax[frame - 1])
        y_axis_frame = y_axis_frame[1:]
        curr_frame = go.Frame(data=[go.Scatter(x=x_axis_frame, y=y_axis_frame, mode='lines')])
        framess.append(curr_frame)

    figure = go.Figure(
        data=[go.Scatter(x=x_axis_framestart, y=y_axis, mode='lines')],
        layout={"title": "Young Adult Decision Scores",
                "updatemenus": [{"type": "buttons",
                                 "buttons": [{"method": "animate",
                                              "label": "play",
                                              "args": [None, {"frame": {"duration": 10,
                                                                        "redraw": True},
                                                              "fromcurrent": False,
                                                              "transition": {"duration": 10,
                                                                             'easing': 'quadratic-in-out'}
                                                              }
                                                       ]
                                              }]
                                 }],
                "xaxis": {"title": "Timepoints [volumes]"},
                # "range":[0,len(df)+5]},
                "yaxis": {"title": "Decision scores from SVM model",
                          "range": [-2, 2]}
                },
        frames=framess)
    #figure.write_html("/content/gdrive/MyDrive/YA_viz_notebook/file.html")
    #figure.show()
    # Plot!
    st.plotly_chart(fig, use_container_width=True)"""



