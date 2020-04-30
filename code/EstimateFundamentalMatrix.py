import cv2
import numpy as np

def estimateFundamentalMatrix(pt1,pt2):
    #Normalize  points
    c1 = np.array([np.mean(pt1[:,0]),np.mean(pt1[:,1])]).reshape(1,2)
    c2 = np.array([np.mean(pt2[:,0]),np.mean(pt2[:,1])]).reshape(1,2)
   
    s1 = np.sqrt(np.sum((pt1 - c1)**2)/2*len(pt1))
    s2 = np.sqrt(np.sum((pt2 - c2)**2)/2*len(pt2))

    normalizedPt1 = (pt1-c1)/s1
    normalizedPt2 = (pt1-c1)/s2

    x1 = normalizedPt1[:,0]; y1=normalizedPt1[:,1]; x2 = normalizedPt2[:,0]; y2 = normalizedPt2[:,1];

    A = np.column_stack((x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,np.ones((8,1))))

    U,S,V = np.linalg.svd(A)
    F = V.T[:,1].reshape(3,3)

    # normalize F
    F = F/np.linalg.norm(F)

    #rank enforcement
    U,S,V = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0
    F = U@S@V
    
    #Denormalize F
    T1 = np.array([[s1, 0, -s1*c1[0,0]],
                   [0, s1, -s1*c1[0,1]],
                   [0, 0, 1]])

    T2 = np.array([[s2, 0, -s2*c2[0,0]],
                   [0, s2, -s2*c2[0,1]],
                   [0, 0, 1]])

    F = T2.T@F@T1
      
    return F


if __name__=='__main__':
    pt1 = np.arange(16).reshape(-1,2)
    pt2 = np.arange(16).reshape(-1,2)
    estimateFundamentalMatrix(pt1,pt2)
     
    
