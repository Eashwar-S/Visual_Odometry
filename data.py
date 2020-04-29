import cv2
import glob
from ReadCameraModel import *
from UndistortImage import *


def readandUndistortImages(path, LUT):
    imgc = []
    names = []
    for filename in glob.glob(path):
        names.append(filename)
    names.sort()

    # Making input video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    count = 1
    for filename in names:
        img1 = cv2.imread(filename, 0)
        color_image = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
        undistortimg1 = UndistortImage(color_image, LUT)
        if count == 1:
            out = cv2.VideoWriter('inputVideo.avi', fourcc, 5.0, (undistortimg1.shape[1], undistortimg1.shape[0]))
            count += 1
        cv2.imshow('image1', undistortimg1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        imgc.append(undistortimg1)
        out.write(undistortimg1)
    out.release()
    return imgc


path = "Oxford_dataset/stereo/centre/*.png"
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')
undistortedImages = readandUndistortImages(path, LUT)

