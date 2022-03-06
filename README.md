# University of Michigan Master of Applied Data Science Capstone Team Brainiac

Authors:

- Stacey Beck 
- Benjamin Merrill
- Mary Soules

## Environment Set-up

- requires path_config.py to access data from cloud storage

### Connect to Dockerfile 
#### build 
docker build -t test_container .

#### run
docker run -p 80:80 -v ~/path/teambrainiac:/source test_container

* specify your path where 'path'


### Install packages locally

--RUN:
- pip install boto3
- pip install nibabel

