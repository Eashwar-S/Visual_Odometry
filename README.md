# Visual_Odometry

In this project have frames of a driving sequence taken by a camera in a car, and the scripts to extract the intrinsic parameters. We will implement the different steps to estimate the 3D motion of the camera, and provide as output a plot of the trajectory of the camera.


### Prerequisite for the code to run
First download the data set neeeded from this [Link](https://drive.google.com/drive/folders/1hAds4iwjSulc-3T88m9UDRsc6tBFih8a). Unzip the data folder and place it in code folder.

Download the input folder from this [link](https://drive.google.com/drive/folders/1bHtRRyyHCVfVULQ9rLM1Tsxr-f1liSLU). Place the input folder in Visual_Odometry folder. After download all the videos the resulting is the structure

- Visual_Odometry

    - code
        - python files
        - Oxford_dataset
    - input
        - inputVideo.avi
    - README

### Run the code

Enter the following to run the code.

For Fundamental Matrix:
```
cd code
python3 EstimateFundamentalMatrix.py
```

For Essential matrix:
```
cd code
python3 EssentialMatrixFromFundamentalMatrix.py
```

To get the fundamental and essential matrix obtained from RANSAC:
```
cd code
python3 GetInliersRansac.py
```

To view the camera trajectory:
```
cd code
python3 camera3DEstimation.py
```

# Result:
The output obtained by comapring our results with Opencv functions:
![](gif/output.gif)

The blue trajectory is got from Opencv functions. The red trajectory is the output got from our functions.


