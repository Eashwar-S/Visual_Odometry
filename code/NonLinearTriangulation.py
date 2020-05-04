import cv2
import numpy

def skew(pt):
    S = np.array([[0, -pt[2], pt[1]],
                  [pt[2], 0, -pt[0]],
                  [-pt[1], pt[0], 0]
                 ])
    return S

def nonLinearTriangulation(K,C1,R1,C2,R2,x1,x2,x0):
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
    pass

