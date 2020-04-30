import cv2
import numpy as np

def extractCameraPose(E):
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]) 
    
    U,S,V = np.linalg.svd(E)
    C1 = U[:,2]; C2 = -U[:,2]; C3 = U[:,2]; C4 = -U[:,2]
    R1 = U @ W @ V; R2 = U @ W @ V; R3 = U @ W.T @ V; R4 = U @ W.T @ V
    
    if np.linalg.det(R1)<0:
        R1=-1*R1
        C1=-1*C1
    if np.linalg.det(R2)<0:    
        R2=-1*R2
        C2=-1*C2
    if np.linalg.det(R3)<0:
        R3=-1*R3
        C3=-1*C3
    if np.linalg.det(R4)<0:
        R4=-1*R4
        C4=-1*C4
   
    return R1,R2,R3,R4,C1,C2,C3,C4


if __name__ == '__main__':
    
    E = np.random.randint(10,size = 9).reshape(3,3)
    R1,R2,R3,R4,C1,C2,C3,C4 = extractCameraPose(E)
    print("function")
    print(R1)
    print(R2)
    print(R3)
    print(R4)
    print(C1)
    print(C2)
    print(C3)
    print(C4)
    print(" ")

    # print("opencv")
    # R1, R2, T = cv2.decomposeEssentialMat(E)
    # print(R1)
    # print(R2)
    # print(T)

