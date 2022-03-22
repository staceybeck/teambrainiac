# University of Michigan Master of Applied Data Science Capstone Team Brainiac

Authors:

- Stacey Beck 
- Benjamin Merrill
- Mary Soules

## Environment Set-up

- requires path_config.py to access data from cloud storage
- store path_config.py in ./source

### Connect to Dockerfile 
#### build 
docker build -t test_container .

#### run
docker run -p 80:80 -v ~/path/teambrainiac:/source test_container

* specify your path where 'path'

### Install packages locally

--RUN:
- pip install boto3 nibabel nilearn

### Data in AWS:
- all_data_dictionary.pkl         : whole brain masked, rt_label filtered, UNNORMALIZED 2d numpy data for all subjects
- whole_brain_all_norm_2d.pkl     : whole brain masked, rt_label filtered, NORMALIZED 2d numpy data for all subjects
- all_data_masksubACC_norm_2d.pkl : Nucleus Accumbens masked, rt_label filtered, NORMALIZED 2d numpy data for all subjects

### All_subject_masked_labeled.ipynb

- This is the pre-processing notebook
- This notebook will perform masking and normalization as well as filtering by label for all matlab data with running one cell. 
- Once the data is returned as masked, filtered and then normalized, check the shape/dims
- Saves the data locally as pickle file

### Access_Load_Data.ipynb

- This is a demo notebook for how the function access_load_data() works in utils.py
- Saves dictionary pickle file of data paths for subject .mat data, subject IDs, mask .mat data, and labels in ./source/data directory

### Mat_to_Numpy.ipynb

- This is a demonstration for how we can access our data from AWS using our data path dictionary pickle file and convert the .mat data to numpy arrays
- Once we access the .mat data in AWS, it downloads locally to .source/data so we can access it and convert to numpy
- Image data is 2d, but we are able to convert to 4d as long as we know the x, y, z components of the image before it was compressed.
- Option to save the 2d, 4d, and label numpy array data in ./source/data directory

### Visualize_Data.ipynb

- Currently unable to run without 4d image data - awaiting original image shape information before this notebook is able to be executed without errors. 



