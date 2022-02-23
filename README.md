# University of Michigan Master of Applied Data Science Capstone Team Brainiac

Authors:
Stacey Beck
Benjamin Merrill
Mary Soules


## Connect to Dockerfile 
### build 
docker build -t test_container .

### run
docker run -p 80:80 -v ~/Desktop/projects/fmri/fmri:/src test_container
