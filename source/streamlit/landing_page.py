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
    'Exploration',
    'Chart Metrics',
]
st.sidebar.header(("Brainiacs"))
st.sidebar.write("Authors: Stacey Rivet Beck"
                 "\n"
                 "\n")
st.sidebar.title('Explore Data')
st.session_state.page_select = st.sidebar.radio('Pages', PAGES)

if st.session_state.page_select == 'Brain Images':
    st.title("Brain States")
    st.sidebar.write("""
            ## Brain Images
             - What is a Voxel? A voxel is a 3D pixel, like a cube! If you were to take a picture of your brain 
               in 3D space, each pixel of that image would represent a voxel. 
            \n
            \n
             - What is fMRI? Functional Magnetic Resonance Imaging. A tube like machine where a person lies flat and still while 
               the machine records brain voxel signals over a period of time - usually while that person is performing some
               sort of visual task on a computer screen inside the machine. 
            \n
            \n
            The images on this page display BOLD (Blood oxygen level dependent) voxel signals in the brain. 
            Areas in yellow/red indicate areas of the brain where
            blood volume is increased and where increases in oxygen exchange occur. Meaning, the brain is active! 
            The blue areas in the brain mean that the blood volume and oxygen exchange decrease, indicating these areas
            are less active.
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

            st.write("This is an interactive brain map. We trained a Support Vector Machine on Young Adult Brains (aged 25 - 27) "
                     "and Adolescent Brains (aged 14 - 16) to predict states of up-regulation and down-regulation - meaning "
                     "an uptake in blood-oxygen volume or decrease in volume during an impulse-task. Voxel intensities that are yellow/red"
                     " are active brain areas during a task and voxel intensities that are dark-blue/light-blue are areas where blood and "
                     "oxygen levels are decreasing."
                     "\n"
                     "In our research, we have found that the Young Adult models predicted better than the Adolescent models. "
                     " You can see clear and distinct voxel clusters in the output from the "
                     " Young Adult model (dark red and blue shades) compared to the Adolescent model output where "
                     "voxel intensities are scattered and dampened (light blue and yellow).")

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
                     "believe to be involved in the impulse-reward system. Each interactive visual is set to the same cut coordinates. We removed"
                     " the grid cross since these areas of the brain are small to visualize. "
                     "\n"
                     "In our Young Adult model, we can see there are large areas of the mPFC that are down-regulating, "
                     "meaning a decrease in activation during the impulse-reward task that the subjects are performing while "
                     " in the magnetic resonance (MR) machine. "
                     "In the Adolescent model, we see more yellow intensities, which imply increased voxel activity (up-regulation) "
                     " in this region.")

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
                "know to be involved in the impulse-reward system and brain disorders. Each interactive visual is set to the same cut coordinates. We removed"
                     " the grid cross since these areas of the brain are small to visualize."
                ""
                "\n"
                " In our Young Adult model, we can see there are large clusters of red voxels in the NAcc, which imply these "
                " subjects are increasing activation during the impulse-reward task that they are performing in while in the MR machine."
                )

    value = st.selectbox("Interactive Brain Visualization. Choose the type of brain activation to view and use your cursor to move the grid cross:", options, format_func=lambda x: display[x])
    get_html(value)

if st.session_state.page_select == "Chart Metrics":
    st.title("Chart Metrics")
    st.sidebar.write("""
            ## Chart Metrics
              - What is a Support Vector Machine? - It is a type of classifier used for non-linear, high dimensional data 
                that tries to separate dissimilar data by defining decision boundaries, which is often created by a hyperplane. 
                (Imagine taking a piece of paper and placing it between an apple and an orange - the paper - like the 
                hyperplane splits the two dissimilar fruits). The classifier tries to maximize the minimal distance 
                between dissimilar data points.
               \n
               \n  
            - What is a decision boundary? - A decision boundary is where the classifier defines whether a data point belongs
              to one class or another. It is defined by vectors that sit 90 degrees from the decision plane. 
            \n
            \n
            The charts on this page discuss decision function scores which represent the distance of important 
            voxels to the classifier's decision boundary. We again compare scores between adolescents and young adults.
    """)

    st.write("(images can be enhanced - click the double arrow at the top right corner of image")
    st.subheader ("Young Adult Whole Brain Mask Model Decision Scores")
    st.write(""
              "\n"
              "\n"
              "Run 2:"
              "\n")
    st.image('/app/teambrainiac/source/streamlit/YA_wb_run2_dfunc_line.png',
             caption=None,
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")
    st.write("Run 3: "
             "\n"
             "\n")
    st.image('/app/teambrainiac/source/streamlit/YA_wb_run3_dfunc_line.png',
             caption=None,
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")
    st.write(" When we trained our young adult models on the support vector machine (SVM) the model output a  "
             "score for each time point (essentially a score for one full 3D brain image). That score represents a  "
             "distance between the important voxels to the classifier's decision boundary. As in, how far away "
             " are we from making a good prediction? "
             ""
             "\n"
             "You can see how the young adult decision scores follow the true labels of the data and do so better "
             " than the adolescent model scores displayed below. This means that the classifier is able to predict t"
             " he brain states in Young Adults.")

    st.subheader("Adolescent Whole Brain Mask Model Decision Scores")
    st.write("\n"
             "\n"
             "Run 2:"
             "\n")
    st.image('/app/teambrainiac/source/streamlit/Adolescent_wb_run2_dfunc_line.png',
             caption=None,
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")
    st.write("Run 3: "
             "\n"
             "\n")
    st.image('/app/teambrainiac/source/streamlit/Adolescent_wb_run3_dfunc_line.png',
             caption=None,
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")
    st.write("You can see how the adolescent decision scores have a hard time following the true labels of the data"
             " in Run 2. However, in Run 3, the decision scores appear to be following the curves of the true labels "
             " better. This suggests that adolescents are learning to regulate their brain activity from one run to "
             " the next while in the scanner, and that the classifier is learning to predict better as a result.")





# Explore page - normalization, voxel feature space
if st.session_state.page_select == 'Exploration':
    st.title("Exploration")
    st.sidebar.write("""
            ## Exploration 
            One of the more interesting aspects of working with functional brain data is visualizing the 
            nature of the data not just as voxel intensities mapped onto a brain template image,
            but as how they exist in time.   
            \n
            \n
            We will take a look at voxel distributions through time as normalized and unnormalized data,  
            as well as how voxels can be represented as features in this dataset. 
            """)
    display = ('Adolescent Detrended Z-score normalization',
               'Adolescent Detrended Percent Signal Change',
               'Young Adult Detrended Unnormalized Data',
               )

    value = 0
    options = list(range(len(display)))
    st.subheader('Voxel Distribution Animation over Time')
    st.write("\n"
             "\n"
             "\n"
             "We took a dive into plotting voxel distributions of our brain data to see "
             " how we could reduce variance - or noise - in our data. We learned that fMRI data, like other"
             " time series data can drift. So, a techinque called 'Detrending' has to be applied to"
             " the series of brain images. We also found that by subtracting our data from the mean over time points"
             " helped to reduce the noise in our data the most."
             "\n"
             "\n"
             "\n"
             "\n")
    st.write("The below voxel distribution animations represent the time series of four separate series (or runs) "
             "of brain scans on a single subject")
    def get_html(value):
        # print(value)
        if value == 0:
            HtmlFile = open("/app/teambrainiac/source/streamlit/ADdtrnd_ZSCORE_normvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=600)

        if value == 1:
            HtmlFile = open("/app/teambrainiac/source/streamlit/ADdtrndpscnormvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=600)

        if value == 2:
            HtmlFile = open("/app/teambrainiac/source/streamlit/YA_dtrnd_Unorm_mvid.html", 'r',
                            encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=600)



    value = st.selectbox("Choose the type of voxel normalization to view:", options, format_func=lambda x: display[x])
    get_html(value)
    st.subheader('Voxel Feature Space')
    st.write("We also wanted to understand our data's feature space"
             " When you think of a data set, maybe you imagine stock prices, or the color of fruit, or even"
             " the number of rooms in a single family home. These data are what we would call the features of "
             " a data set. In brain data, the features are not well defined and are more abstract. They are the "
             " voxel intensities. "
             "\n"
             "\n"
             "Below we have plotted the voxel signals of whole brain data over the course of time."
             " In this data set, we have approximately 84 time points along the x-axis and "
             "  3D brain data (represented in 2D) along the y - axis. Each time point is plotting "
             " approximately 238,000 voxel values for a single person. They look like straight lines, which is "
             " a strange concept when you think about human brains!"
             "\n")
    st.image("/app/teambrainiac/source/streamlit/feature_space.png",
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")


if st.session_state.page_select == 'Tables':
    st.title("Tables")
    st.sidebar.write("""
            ## Tables
            The tables on this page provide use with useful information about our decision choices for further
            preprocessing of our data. 
            """)
    st.image("/app/teambrainiac/source/streamlit/newplot.png",
             width=None,
             use_column_width=None,
             clamp=False,
             channels="RGB",
             output_format="auto")
