# University of Michigan Master of Applied Data Science Capstone 
## Using Support Vector Machine and 3D Convolutional Neural Networks to Predict Brain States in Task-Based Functional Magnetic Resonance Imaging (fMRI)

Authors:

- Stacey Beck 
- Benjamin Merrill
- Mary Soules


This README contains the following sections:
-   [Project Description](https://github.com/yecatstevir/teambrainiac#project-description)
-	[Repository Organization](https://github.com/yecatstevir/teambrainiac#repository-organization)
-	[Environment Requirements](https://github.com/yecatstevir/teambrainiac#environment-requirements)
    - [Connect to Docker](https://github.com/yecatstevir/teambrainiac#connect-to-dockerfile)
    - [Streamlit set-up](https://github.com/yecatstevir/teambrainiac#for-streamlit-app)
-   [Single Subject](https://github.com/yecatstevir/teambrainiac#single-subject)
-   [Adolescent and Young Adult](https://github.com/yecatstevir/teambrainiac#adolescent-and-young-adult)
-   [Deep Learning](https://github.com/yecatstevir/teambrainiac#deep-learning)
  

## Project Description

This project is part of an extension of research through the University of Michigan Medicine department that studies substance use disorders through real-time fMRI neurofeedback analyses. The study has recruited over 80 voluntary subjects to participate in the study. We are authorized to share our process and results from our study but unable to provide access to the data. All data has been de-identified and consent given to share results. Please contact the authors directly for a demonstration on how the code runs. 

In our study, we used temporal 3D brain data from 52 healthy adolescents and young adults ranging from the ages of 16 years to 21 years old. We split out analyses up by subject as well as group for the SVM analyses and by group for the deep learning analyses.  

## Repository Organization
    ├── Dockerfile                                   <- Details to build and run Docker container
    ├── requirements.txt                             <- Installed dependencies by Docker
    ├── README.md                                    <- This README file
    ├── .gitignore                                   <- Specifies files ignored by git
    |
    └── source/
    |    ├── streamlit/
    |    |    └──                                    <- Contains landing_page.py app
    |    ├── helper/
    |    |    └──                                    <- Contains modules and Jupyter Notebooks from early project exploration
    |    |__ SingleSubjectSVM_Norm_CV.ipynb          <- Contains modules to test normalization strategies (no normalization, percent signal change,z
    |    |                                              normalization) and to run a cv search on best strategy once chosen.
    |    |__ BuildSingleSubjectSVM_Models.ipynb      <- Contains modules to run single subject SVM model. Output can be used inline or saved for future
    |    |                                              use.
    |    |__ DataExplorationNotebook_SingleSubjectSVM.ipynb  <- Contains modules to explore normalization strategies we employed and to look at cross
    |    |                                                      validation results. This notebook pulls in previously stored data after running the XXXX
    |    |__ VisualizationPlayground.ipynb           
    |    |__ single_subject.py                       <- Contains functions to access data, mask data, normalize data, run single subject model. The model
    |    |                                              will run on more than one turn for training, if desired. At this point testing is done on single
    |    |                                              runs only. This also contains functions for getting predictions to be stored for later use,
    |    |                                              accuracy scores for data exploration.
    |    |__ brain_viz_single_subj.py                <- Contains functions to create bmaps for brain visualizations, functions for brain images,
    |    |                                              interactive brain images, and functions to display decision functions scores across the
    |    |                                              timeseries
    |    ├── group_svm/                            
    |    |__  |__ data/                           
    |    |__  |__ images/                          
    |    |__  |__ Adolescent_Group_SVM.ipynb         <- Adolescent model training, brain and decision score visualization
    |    |__  |__ Explore_data.ipynb                 <- Normalization notebook plotting voxel histograms per subject
    |    |__  |__ Young_Adult_Group_SVM.ipynb        <- Young Adult model training, brain and decision score visualization
    |    |__  |__ Timeseries_Cross_Validation.ipynb  <- Cross validation used to determine classifier parameters 
    |    |__  |__ Group_metrics.ipynb                <- Generates plots group-level metrics 
    |    |__  |__ Statistical_tests_Group.ipynb      <- Statistical tests for beta maps at group level
    |    |__  └── access_data.py                     <- Connect to AWS, uploads and downloads data
    |    |__  └── analysis.py                        <- Collects metrics from models and saves data, uploading to AWS
    |    |__  └── cross_validation.py                <- Partitions data using TimeSeries package from Sklearn for cross validation and gridsearch
    |    |__  └── process.py                         <- Processes the data from MATLAB further, organizes data for model training
    |    |__  └── train.py                           <- Training file for SVM using Train, Validation, Test or Train and Test sets
    |    |__  └── visualize.py                       <- Code to visualize certain plots using Nilearn as well as normalization exploration
    |    └── data/   
    |    |    └──                                    <- Contains data needed to be accessed within /source. Data dictionary, and T1 images for
    |    |                                              visualization 
    |    ├── DL/                                     <- Deep Learning folder containing 3 main scripts to run 3D-CNN from AWS to analysis
    |    |    └── PreprocessToAws.ipynb              <- From Matlab inputs to pytorch-compatible tensor files
    |    |    └── Group3DCNN.ipynb                   <- Loads tensor training, validation, and testing for model building
    |    |    ├── metrics/
    |    |    |    └── VisualizationCreation.ipynb   <- Analyzes metrics from Group3DCNN.ipynb file


## Environment Requirements

- requires path_config.py to access data from cloud storage
- store path_config.py in the directory from which you are running the notebooks and scripts

### Connect to Dockerfile 
#### build 
    docker build -t test_container .

#### run
    docker run -p 80:80 -v ~/path/teambrainiac:/source test_container

* specify your path where 'path'

### Install packages locally

    !pip install boto3 nibabel nilearn
    
### For Streamlit app
    Streamlit's prerequisites:
        - IDE or text editor
        - Python 3.7 - Python 3.9
        - PIP
        
#### macOS/Linux:
        sudo easy_install pip
        
        pip3 install pipenv
        
        pipenv shell
        
        pipenv install streamlit
        
#### Run app in repo:
        pipenv shell
        
        streamlit run landing_page.py
        
#### To view the app in browser:
        https://share.streamlit.io/yecatstevir/teambrainiac/main/source/streamlit/landing_page.py
        

### Single Subject


### Adolescent and Young Adult


### Deep Learning
The deep-learning approach to group level analysis is an attempt to see if we can provide better predictions to brain-states than Support Vector Machines. The preprocessing, model building and training, and visualizations can be found in the DL folder in source. It is split into three main notebooks. PreprocessToAws.ipynb outlines the preprocessing, data normalization, and formatting for the Pytorch dataloader. Group3DCNN.ipynb creates the model and has scripts for training, validation, and testing. VisualizationCreation.ipynb is in the metrics folder and uses metrics returned by the model and creates meaningful insights from the neural network. All .py files contain helper functions for these three notebooks, and the metrics folder contains some csv metric files, a few trained CNN models, and visualizations.


### Data in AWS:
- ADD THE subject folder descriptions
- single_subj_T1_resampled.nii    : NIFTI T1 Brain Image file of a single subject for Visualizations
- w3rtprun_01.nii                 : Data Affine for maping voxel coordinates to Real World Coordinates for Visualizations

