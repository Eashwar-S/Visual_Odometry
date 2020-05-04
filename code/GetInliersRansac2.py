from EstimateFundamentalMatrix import * 
from featureMatching import *
from EssentialMatrixFromFundamentalMatrix import *
import cv2

def sampleRandomFeatures(f1,f2):
    rand = np.random.randint(len(f1),size = 8)
    sampledF1 = np.array([f1[rand[i]] for i in range(8)])
    sampledF2 = np.array([f2[rand[i]] for i in range(8)]) 
    return sampledF1,sampledF2

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


def getInliersRansac(matches,kp1,kp2): 
    listKp1,listKp2 = getKeyPointCoordinates(matches,kp1,kp2)

    bestF = None; bestInliersKp1 = None; bestInliersKp2 = None; bestCount= 0;
     

    for i in range(500):
        S = []
        count = 0
        f1 ,f2 = sampleRandomFeatures(listKp1,listKp2)
        F = estimateFundamentalMatrix(f1,f2) 
        Sx = []
        Sy = []

        x1 = np.hstack((listKp1,np.ones((len(listKp1),1))))
        x2 = np.hstack((listKp2,np.ones((len(listKp1),1))))
        e1, e2 = x1 @ F.T, x2 @ F
        dist = np.sum((x2@F)* x1, axis = 1, keepdims=True)**2 /   \
               np.sum(np.hstack((e1[:, :-1],e2[:,:-1]))**2, axis = 1, keepdims=True)
        
        inliers = dist<= 0.07
        inliersIndex = np.argwhere(inliers==True) 
        inliersKp1 = x1[inliersIndex[:,0]]
        inliersKp2 = x2[inliersIndex[:,0]]
        
        count = np.sum(inliers)
        if bestCount <  count:
            bestInliersKp1 = inliersKp1
            bestInliersKp2 = inliersKp2

            bestCount = count
            bestF = F 

    return bestF,bestInliersKp1,bestInliersKp2



if __name__=='__main__':
    img1 = cv2.imread('./Oxford_dataset/stereo/centre/1399381447017075.png')
    img2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381447079564.png')
    matches,kp1,kp2,des1,des2 = featureMatching(img1,img2)

    K = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3,3)
    F,inliersP1,inliersP2 = getInliersRansac(matches,kp1,kp2)
    E = essentialMatrix(K,F)
    print("function")
    print(F)
    print(E)

    listKp1,listKp2 = getKeyPointCoordinates(matches,kp1,kp2)
    Fcv, inliers = cv2.findFundamentalMat(listKp1,listKp2,method=cv2.FM_RANSAC)
    Ecv,_ = cv2.findEssentialMat(listKp1,listKp2,cameraMatrix=K,method=cv2.RANSAC)
    print("Opencv")
    print(Fcv)
    print(Ecv)
    cv2.waitKey()



