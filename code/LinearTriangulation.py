import cv2
import numpy as np

def skew(pt):
    S = np.array([[0, -pt[2], pt[1]],
                  [pt[2], 0, -pt[0]],
                  [-pt[1], pt[0], 0]
                 ])
    return S

def linearTriangulation(K:np.array,C1:np.array,R1:np.array,C2:np.array,R2:np.array,pts1:list,pts2:list)->np.array:
    C1 = C1.reshape(-1,1); C2 = C2.reshape(-1,1)
    x = np.zeros((len(pts1),3)) 
    P1 = K@np.hstack((R1,-R1@C1)) 
    P2 = K@np.hstack((R2,-R2@C2)) 
         
    for i in range(len(pts1)):
        s1 = skew(pts1[i])
        s2 = skew(pts2[i])
        A = np.vstack((s1@P1,s2@P2))
        U,S,V = np.linalg.svd(A)
        P = V[-1]
        if(P[-1]!=0):
            P = P/P[-1]
        x[i] = P.reshape(-1)[0:3]
    return x 


if __name__=='__main__':
    K = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3,3)
    C1 = np.array([0,0,0])
    C2 = np.array([0,0,0])
    R1 = np.eye(3)
    R2 = np.eye(3)
    pt1 = np.array([[1,2,3]]) 
    pt2 = np.array([[1,2,3]]) 
    x=linearTriangulation(K,C1,R1,C2,R2,pt1,pt2)
    print("method")
    print(x)

    C1 = C1.reshape(-1,1); C2 = C2.reshape(-1,1)
    pt1 = np.array([[2],[6]]) 
    pt2 = np.array([[3],[8]]) 
    P1 = np.hstack((R1,C1)) 
    P2 = np.hstack((R2,-R2@C2)) 
    P=cv2.triangulatePoints(P1,P2,pt1,pt2)
    print("opencv")
    print(P)
    # print(P/P[-1])

     

