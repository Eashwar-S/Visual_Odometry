from ReadCameraModel import *
from featureMatching import *
from EstimateFundamentalMatrix import *
from GetInliersRansac2 import *
from ExtractCameraPose import * 
from LinearTriangulation import *
from DisambiguateCameraPose import * 
import matplotlib.pyplot as plt

import cv2 


def intrinsicMatrix(fx,fy,cx,cy,s):
    K = np.array([[fx,s,cx],
                  [0,fy,cy],
                  [0,0,1]])
    return K

def getKeyPointCoordinates(matches,kp1,kp2):
    listKp1 = []
    listKp2 = []

    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1Idx = mat.queryIdx
        img2Idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1Idx].pt
        (x2, y2) = kp2[img2Idx].pt

        # Append to each list
        listKp1.append((x1, y1))
        listKp2.append((x2, y2))
    
    listKp1 = np.array(listKp1)
    listKp2 = np.array(listKp2)
    return listKp1,listKp2


def main():
    fig,ax = plt.subplots()
    result = []

    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')
    K = intrinsicMatrix(fx,fy,cx,cy,0)

    cap = cv2.VideoCapture('../input/inputVideo.avi')
    frame2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381444704913.png')
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) 
     
    Ht = np.eye(4)
   
    frameNumber = 0
    while(cap.isOpened()):
       frame1  = frame2
       _, frame2 = cap.read()

       if frame2 is None:
           break

       frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) 
       
       #####################################################################
       #                    USING CREATED FUNCTIONS
       #####################################################################
       matches,kp1,kp2,des1,des2 = featureMatching(frame1,frame2)
       F,inliersP1,inliersP2 = getInliersRansac(matches,kp1,kp2)


       listKp1,listKp2 = getKeyPointCoordinates(matches,kp1,kp2)
       E = estimateFundamentalMatrix(K,F)
       # E,_ = cv2.findEssentialMat(listKp1,listKp2,cameraMatrix=K,method=cv2.RANSAC)
       R1,R2,R3,R4,C1,C2,C3,C4  = extractCameraPose(E)

        
       Xset1 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C1,R1,inliersP1,inliersP2)
       Xset2 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C2,R2,inliersP1,inliersP2)
       Xset3 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C3,R3,inliersP1,inliersP2)
       Xset4 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C4,R4,inliersP1,inliersP2)
       Xset = [Xset1,Xset2,Xset3,Xset4] 
       Cset = [C1,C2,C3,C4]
       Rset = [R1,R2,R3,R4]
       C,R = disambiguateCameraPose(Cset,Rset,Xset)
       
       #####################################################################
       #                      USING OPENCV
       #####################################################################
       # matches,kp1,kp2,des1,des2 = featureMatching(frame1,frame2)
       # listKp1,listKp2 = getKeyPointCoordinates(matches,kp1,kp2)
       # Ecv,_ = cv2.findEssentialMat(listKp1,listKp2,cameraMatrix=K,method=cv2.RANSAC)
       # points, R, C, mask = cv2.recoverPose(Ecv, listKp1, listKp2,cameraMatrix=K)

       if C is None or R is None:
           continue
    
       ####################################################################
       #                     HOMOGENOUS TRANSFORMATION
       ####################################################################
       Htn = np.hstack((R,C)) 
       Htn = np.vstack((Htn,[0,0,0,1]))
       Ht1 = Ht@Htn
       xt1 = Ht1[0,3] 
       zt1 = Ht1[2,3]
       Ht = Ht1 
       

       #####################################################################
       #                      RESULTS
       ####################################################################
       #---------------------------------
       # results for created functions
       #---------------------------------
       result.append([-xt1,zt1])
       ax.plot([-xt1],[zt1],'o')
      
       #---------------------------------
       #results for opencv
       #---------------------------------
       # result.append([-xt1,zt1])
       # ax.plot([-xt1],[zt1],'o')
      

       plt.pause(0.01)

       frameNumber += 1 
       print(frameNumber)

       cv2.imshow('img2',frame1)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
    ########################################################################
    #                       SAVE RESULTS
    #######################################################################
    #------------------------------
    #     created functions
    #------------------------------
    np.savetxt("result.csv", result, delimiter=",") 

    #------------------------------
    #     opencv
    #------------------------------
    # np.savetxt("resultcv.csv", result, delimiter=",") 

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()
    






