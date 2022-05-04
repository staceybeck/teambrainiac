
# Using Support Vector Machine and 3D Convolutional Neural Networks to Predict Brain States in Task-Based Functional Magnetic Resonance Imaging (fMRI)
## University of Michigan Master of Applied Data Science Capstone 

**Authors:**

[**Stacey Beck**](mailto:starbeck@umich.edu)

[**Benjamin Merrill**](mailto:benme@umich.edu)

[**Mary Soules**](mailto:mfield@umich.edu)

For details on our approaches and findings, check out our report: [Report](https://github.com/yecatstevir/teambrainiac/blob/main/Capstone_Final_Report.pdf)

Check out our interactive brain maps, graphs and descriptions of what we looked at in this study: [Landing Page](https://share.streamlit.io/yecatstevir/teambrainiac/main/source/streamlit/landing_page.py)


This README contains the following sections:
-   [Project Description](https://github.com/yecatstevir/teambrainiac#project-description)
-	[Repository Organization](https://github.com/yecatstevir/teambrainiac#repository-organization)
-	[Environment Requirements](https://github.com/yecatstevir/teambrainiac#environment-requirements)
    - [Connect to Docker](https://github.com/yecatstevir/teambrainiac#connect-to-dockerfile)
    - [Streamlit set-up](https://github.com/yecatstevir/teambrainiac#for-streamlit-app)
-   [Single Subject](https://github.com/yecatstevir/teambrainiac#single-subject)
-   [Adolescent and Young Adult](https://github.com/yecatstevir/teambrainiac#adolescent-and-young-adult)
-   [Deep Learning Notebook](https://github.com/yecatstevir/teambrainiac#deep-learning-notebook)
  

## Project Description

We are collaborating with the University of Michigan Medicine Psychology Research Department under the research of Principal Investigator Dr. Meghan Martz, PhD. This opportunity was presented to us by one of our authors, Mary Soules, who works as an Application Programming Analyst Senior in the UM research lab processing and analyzing brain images captured through Dr. Martz' study of substance use disorders through real-time fMRI neurofeedback analyses. The study has recruited over 80 voluntary subjects to participate in the study. We are authorized to share our process and results from our study but unable to provide access to the data. All data has been de-identified and consent given to share results. Please contact the authors directly for a demonstration on how the code runs. 

For this project, we used temporal 3D brain data from 52 healthy adolescents (14 - 16 years old) and young adults (25 - 27 years old). We split our analyses up by subject as well as group for the SVM analyses and by group for the deep learning analyses.  

We have three approaches to looking at the fMRI data captured through this study: 
- First, single subject SVM to look at individual differences in brain activation and metrics associated within a single subject. This approach would allow us to use trained models at the individual level to help in personalizing treatment for addiction to help individuals up or down-regulate areas in the brain that could be playing a role in their addiction. 
- Second, a group-level SVM approach to study whether we can predict brain states or locate other regions of interest other than the Nucleus Accumbens on a group level and apply it across subjects, as well as understand if differences exist between groups. 
- Third, a deep-learning approach to group level analysis to see if we can provide better predictions to brain-state data. 


## Repository Organization
    ├── Dockerfile                                   <- Details to build and run Docker container
    ├── requirements.txt                             <- Installed dependencies by Docker
    ├── README.md                                    <- This README file
    ├── .gitignore                                   <- Specifies files ignored by git
    |
    └── source/
    |    ├── streamlit/
    |    |    └──                                    <- Contains landing_page.py app
    |    |__single_subject/
    |    |__  |__ SingleSubjectSVM_Norm_CV.ipynb          <- Contains modules to test normalization strategies (no normalization, percent signal change,z
    |    |    |                                              normalization) and to run a cv search on best strategy once chosen. Will also run visualizations if don't want to store.
    |    |__  |__ BuildSingleSubjectSVM_Models.ipynb      <- Contains modules to run single subject SVM model. Output can be used inline or saved for
    |    |    |                                              future use.
    |    |__  |__SingleSubjSVM_Analysis.ipynb               <-Contains modules to load and run Single Subject Models that have been stored locally.                           
    |    |__  |__ DataExplorationNotebook_SingleSubjectSVM.ipynb  <- Contains modules to explore normalization strategies we employed and to look at cross
    |    |    |                                                      validation results. This module loads data stored locally.
    |    |__  |__DataExplorationofSingleSubject_SVM.ipynp <-Contains modules to explore normalization and cross validations while also running visualizations
    |    |__  |__ single_subject.py                       <- Contains functions to access data, mask data, normalize data, run single subject model. The
    |    |    |                                              model will run on more than one turn for training, if desired. At this point testing is done
    |    |    |                                              on single  runs only. This also contains functions for getting predictions to be stored for       |    |    |                                              later use, accuracy scores for data exploration.
    |    |    |                                          
    |    |    |                                          
    |    |__  |__brain_viz_single_subj.py                <- Contains functions to create bmaps for brain visualizations, functions for brain images,
    |    |                                                  interactive brain images, and functions to display decision functions scores across the
    |    |                                                  timeseries
    |    ├── group_svm/                            
    |    |__  |__ data/                           
    |    |__  |__ images/                          
    |    |__  |__ Adolescent_Group_SVM.ipynb         <- Adolescent model training, brain and decision score visualization
    |    |__  |__ Explore_data.ipynb                 <- Normalization notebook plotting voxel histograms per subject
    |    |__  |__ Young_Adult_Group_SVM.ipynb        <- Young Adult model training, brain and decision score visualization
    |    |__  |__ Timeseries_Cross_Validation.ipynb  <- Cross validation used to determine classifier parameters 
    |    |__  |__ Group_metrics.ipynb                <- Generates plots group-level metrics 
    |    |__  |__ Statistical_tests_Beta_maps.ipynb      <- Statistical tests for beta maps at group level
    |    |__  |__ Statistical_tests_decisions.ipynb      <- Statistical tests for decision scores at group level
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
    |    |__  └── PreprocessToAws.ipynb              <- From Matlab inputs to pytorch-compatible tensor files
    |    |__  └── Group3DCNN.ipynb                   <- Loads tensor training, validation, and testing for model building
    |    |__  ├── metrics/
    |    |__  |    └── VisualizationCreation.ipynb   <- Analyzes metrics from Group3DCNN.ipynb file


## Environment Requirements

- requires path_config.py to access data from cloud storage
- store path_config.py in the directory from which you are running the notebooks and scripts


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
        
#### [Click here to view our app in your browser](https://share.streamlit.io/yecatstevir/teambrainiac/main/source/streamlit/landing_page.py)

### Preprocessing Pipeline

<img src="https://github.com/yecatstevir/teambrainiac/blob/main/source/streamlit/preprocessing_pipeline.PNG" alt="Preprocessing Pipeline" title="Preprocessing Pipeline">

### Single Subject
The single subject analysis approach aims to explore whether SVM can be used to train individual models to predict brain states in regulating reward circuity in the brain. This approach can be used to explore whether there are differences between individuals that alter how well the model can predict and two whether or not other areas of the brain beyond the nucleus accumbens are involved in how well a person is able to up or down regulate their reward system. The nucleus accumbens is thought to be a key driver for reward regulation, but individuals could have unique regions that help them regulate their reward state. Below is the notebook to run the single subject model. It has the ability to access the data in cloud, process the data (masking, filtering by label, train on the whole brain, the 4 submasks, and the 4 regions of interest, save the data and metrics locally and visualize the brain mask, run roc curves, confusion matrix, and decision scores.

[Single Subject SVM Notebook](source/single_subject/BuildSingleSubjectSVM_Models.ipynb)

### Adolescent and Young Adult
The group analysis approach aims to explore differences within groups and between groups in regulating the reward circuitry in the Nucleus Accumbens and to see if other regions of interest are involved in this regulation. Below are the two notebooks that split each analyses. Due to the length of training for each notebook, in order to train on the whole brain, 5 submasks and 4 regions of interests, each model training is run one at a time. The notebook is able to access the data in cloud storage, process the data (masking, filtering by label, split and concatenate the data) and train/ save the model locally, save the metrics to the cloud and visualize the brain mask as well as a few key regions of interest, visualize the decision function scores and histogram. 

[Adolescent Notebook](source/group_svm/Adolescent_Group_SVM.ipynb)

[Young Adult Notebook](source/group_svm/Young_Adult_Group_SVM.ipynb)

Other notebooks in this directory used as a part of these analyses include normalization exploration, cross-validation, metrics analysis, and statistical analyses of beta maps as well as decision function scores. Each notebook defines the variables of interest to run (such as group type, mask type, runs to train and test, normalization to apply) and uses function calls to various modules stored in this directory for a more clean presentation. 


### Deep Learning Notebook
The deep-learning approach to group level analysis is an attempt to see if we can provide better predictions to brain-states than Support Vector Machines. The preprocessing, model building and training, and visualizations can be found in the DL folder in source. It is split into three main notebooks. PreprocessToAws.ipynb outlines the preprocessing, data normalization, and formatting for the Pytorch dataloader. Group3DCNN.ipynb creates the model and has scripts for training, validation, and testing. VisualizationCreation.ipynb is in the metrics folder and uses metrics returned by the model and creates meaningful insights from the neural network. All .py files contain helper functions for these three notebooks, and the metrics folder contains some csv metric files, a few trained CNN models, and visualizations.

[Preprocessing Notebook](source/DL/PreprocessToAws.ipynb)

[Modeling Notebook](source/DL/Group3DCNN.ipynb)

[Run the Visualizations Here](source/DL/metrics/VisualizationCreation.ipynb)

### Information only
All pre-processing of the data was done outside of the repository. The following software packages were used for pre-processing: MATLAB, SPM (Statistical Parametric Modulation). Automated Anatomical Labeling (AAL) (or Anatomical Automatic Labeling) is a software package that captures the digital atlas of the brain. The following steps were used to create the pre-processed .mat files. Preprocessing included using pre-created preprocessing pipelines Mary Soules had built in SPM for fMRI research. Preprocessing steps included: slicetiming correction, realignment of scans, structural images coregistered to functional images, warping of high-resolution structural image to common MNI space using MNI152 image, warping functional images to common MNI space. All quality control of images used in analysis were performed prior to uploading to AWS. Quality control measures included checking coregistered structural images and functional images to make sure they were in the same space, checking warped structural images and warped functional images against the MNI template supplied to make sure all images were in the correct space, checked motion by running a script that looks at amount of motion in all runs and threw out individuals that exceeded our threshold for motion. Once the final dataset was processed, a MATLAB function was used to flatten the 4-D images to a 2-D matrix for each run and saved these data in a .mat file that could be read from python.
