from EstimateFundamentalMatrix import * 
from featureMatching import *
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
    n = 0 
    bestF = None 
    inliersP1 = []
    inliersP2 = []
    for _ in range(20):
        f1 ,f2 = sampleRandomFeatures(listKp1,listKp2)
        F = estimateFundamentalMatrix(f1,f2) 
        Sx = []
        Sy = []
        for j in range(len(listKp1)):
            x1 = np.append(listKp1[j],1).reshape(3,1)
            x2 = np.append(listKp2[j],1).reshape(3,1)
            dist = x2.T@F@x1 
            if (np.max(dist) < 0.01):
                Sx.append(x1.reshape(-1))
                Sy.append(x2.reshape(-1))

        if n < len(Sx):
            n = len(Sx)
            bestF = F
            inliersP1 = Sx
            inliersP2 = Sy
    return bestF,inliersP1,inliersP2



if __name__=='__main__':
    img1 = cv2.imread('./Oxford_dataset/stereo/centre/1399381447017075.png')
    img2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381447079564.png')
    matches,kp1,kp2,des1,des2 = featureMatching(img1,img2)
    F,inliersP1,inliersP2 = getInliersRansac(matches,kp1,kp2)
    print("function")
    print(F)

    listKp1,listKp2 = getKeyPointCoordinates(matches,kp1,kp2)
    F, inliers = cv2.findFundamentalMat(listKp1,listKp2,method=cv2.FM_RANSAC)
    print("Opencv")
    print(F)
    cv2.waitKey()



