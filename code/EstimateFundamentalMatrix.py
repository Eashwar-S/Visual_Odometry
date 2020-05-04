import cv2
import numpy as np


def estimateFundamentalMatrix(pt1:np.array,pt2:np.array)->list:
    N = len(pt1)

    #Denoise data(scale and translate)
    c1 = np.sum(pt1,axis = 0)/N                   #Compute centroid 
    c2 = np.sum(pt2,axis = 0)/N                   #Compute centroid 
    dc = np.sqrt(np.sum((pt1 - c1)**2,axis = 1))  # distance of points from centroid
    davg = np.sum(dc)/N                           # average distance
    s1 = np.sqrt(2)/davg                          # scale average dist to sqrt(2)

    dc = np.sqrt(np.sum((pt2 - c2)**2,axis = 1))  # distance of points from centroid
    davg = np.sum(dc)/N                           # average distance
    s2 = np.sqrt(2)/davg                          # scale average dist to sqrt(2)
    
   
    normalizedPt1 = (pt1-c1)*s1                   # Normalize the points
    normalizedPt2 = (pt2-c2)*s2                   # Normalize the points
    
   
    x1 = normalizedPt1[:,0]; y1=normalizedPt1[:,1]; x2 = normalizedPt2[:,0]; y2 = normalizedPt2[:,1];

    # A = np.column_stack((x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y2,np.ones((N,1))))
    A = np.column_stack((x2*x1, x2*y1, x2, y2 * x1, y2 * y1, y2, x1,  y1, np.ones((N,1))))	

    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)                        #compute fundamental Matrix

    U,S,V = np.linalg.svd(F)                      #rank enforcement
    S = np.diag(S)
    S[2,2] = 0
    F = U@S@V
    
    T1 = np.array([[s1, 0, -s1*c1[0]],            
                   [0, s1, -s1*c1[1]],
                   [0, 0, 1]])

    T2 = np.array([[s2, 0, -s2*c2[0]],
                   [0, s2, -s2*c2[1]],
                   [0, 0, 1]])

    F = T2.T@F@T1                                #Denormalize F
    
    F = F/F[2,2]
    return F


if __name__=='__main__':
    pt1 = np.random.randint(20,size=20).reshape(-1,2)
    pt2 = np.random.randint(16,size=20).reshape(-1,2)
    F =estimateFundamentalMatrix(pt1,pt2)
    print("method")
    print(F)
    print("opencv")
    Fcv,_ = cv2.findFundamentalMat(pt1,pt2,cv2.FM_8POINT)
    print(Fcv)

   


    
