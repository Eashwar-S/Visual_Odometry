import cv2
import numpy as np


def essentialMatrix(K,F):
    E = K.T@F@K
    U,S,V = np.linalg.svd(E)
    S = [1,1,0]
    S = np.diag(S) 
    
    E = U@S@V
    E = E/np.linalg.norm(E)
    return E 
   

if __name__ == '__main__':
    K = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3,3)
    F = np.arange(9).reshape(3,3)
    E = essentialMatrix(K,F)
    print(E)

