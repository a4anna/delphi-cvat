# delphi-cvat
Integrating Delphi and CVAT

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.7, Pytorch 1.5, CUDA 10.2, GTX 1080 GPUs

- Clone the repository 
```
git clone https://github.com/a4anna/delphi-cvat && cd delphi-cvat
export DELPHI=$PWD
```
- Setup python environment
```
conda env create -f environment.yml
conda activate delphi
```
- Set environment variable
```
export CVAT_USER={CVAT-USERNAME}
export CVAT_PASS={CVAT-PASSWORD}
export PYTHONPATH=$DELPHI:$PYTHONPATH
```
## Data Directory setup
+data-root  
  +labeled/  
   +0/ # labeled negative image ddirectory  
    -000.jpg  
    -501.jpg  
    -\*.jpg     
   +1/ # labeled positive image ddirectory  
    -011.jpg  
    -203.jpg  
    -\*.jpg  
  +unlabeled/  
   -\*.jpg  
   
## Generate proto files
```
 cd $DELPHI
 python generate_proto.py
```
## Run

#### Launch OpenCV CVAT
#### Modify config.yml
#### Start Delphi 
```
./run.sh
``` 

