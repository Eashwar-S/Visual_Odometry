from ReadCameraModel import *
from featureMatching import *
from EstimateFundamentalMatrix import *
from GetInliersRansac import *

import cv2 


def intrinsicMatrix(fx,fy,cx,cy,s):
    K = np.array([[fx,0,cx],
                  [0,fy,cy],
                  [0,0,1]])
    return K



def main():
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')
    K = intrinsicMatrix(fx,fy,cx,cy,0)

    cap = cv2.VideoCapture('../input/inputVideo.avi')
    while(cap.isOpened()):
       _, frame1 = cap.read()
       _, frame2 = cap.read()
       

       matches,kp1,kp2,des1,des2 = featureMatching(frame1,frame2)



       cv2.imshow('img1',frame1)
       cv2.imshow('img2',frame2)
       if cv2.waitKey(0) & 0xFF == ord('q'):
           break

    cap.release()
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()
    






