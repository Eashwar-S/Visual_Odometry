from ReadCameraModel import *
from featureMatching import *
from EstimateFundamentalMatrix import *
from GetInliersRansac import *
from ExtractCameraPose import * 
from LinearTriangulation import *
from DisambiguateCameraPose import * 
import matplotlib.pyplot as plt

import cv2 


def intrinsicMatrix(fx,fy,cx,cy,s):
    K = np.array([[fx,0,cx],
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

    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')
    K = intrinsicMatrix(fx,fy,cx,cy,0)

    cap = cv2.VideoCapture('../input/inputVideo.avi')
    frame2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381444704913.png')
    
    Ht = np.eye(4)
    while(cap.isOpened()):
       frame1  = frame2
       _, frame2 = cap.read()
       
       matches,kp1,kp2,des1,des2 = featureMatching(frame1,frame2)
       # F,inliersP1,inliersP2 = getInliersRansac(matches,kp1,kp2)
       # E = estimateFundamentalMatrix(K,F)
       # R1,R2,R3,R4,C1,C2,C3,C4  = extractCameraPose(E)
        
       # Xset1 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C1,R1,inliersP1,inliersP2)
       # Xset2 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C2,R2,inliersP1,inliersP2)
       # Xset3 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C3,R3,inliersP1,inliersP2)
       # Xset4 = linearTriangulation(K,np.zeros((3,1)),np.eye(3),C4,R4,inliersP1,inliersP2)
       # Xset = [Xset1,Xset2,Xset3,Xset4] 
       # Cset = [C1,C2,C3,C4]
       # Rset = [R1,R2,R3,R4]
       # C,R = disambiguateCameraPose(Cset,Rset,Xset)
       

       #Using opencv
       listKp1,listKp2 = getKeyPointCoordinates(matches,kp1,kp2)
       E,_ = cv2.findEssentialMat(listKp1,listKp2,cameraMatrix=K,method=cv2.RANSAC)
       # points, R, C, mask = cv2.recoverPose(E, listKp1, listKp2,cameraMatrix=K)
       points, R, C, mask = cv2.recoverPose(E, listKp1, listKp2)
        
       # M_r = np.hstack((R, t))
       # M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

       # P_l = np.dot(K,  M_l)
       # P_r = np.dot(K,  M_r)
       # point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))
       # point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
       # point_3d = point_4d[:3, :].T

       Htn = np.hstack((R,C)) 
       Htn = np.vstack((Htn,[0,0,0,1]))
       Ht1 = Ht@Htn
       xt1 = Ht1[0,3] 
       zt1 = Ht1[2,3]
       Ht = Ht1 
       
       print(R)
       print(C)
       ax.plot([-xt1],[zt1],'o')
       plt.pause(0.01)

       cv2.imshow('img2',frame2)
       match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:50], None)
       cv2.imshow('Matches', match_img)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # plt.show()
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()
    






