import cv2
import numpy as np
 
def essentialMatrix(K,F):
    E = K.T@F@K
    U,S,V = np.linalg.svd(E)
    # S = [1,1,0]
    S = np.diag(S) 
    S[-1,-1]= 0
    E = U@S@V
    # E = E/np.linalg.norm(E)
    return E 
   
if __name__ == '__main__':
    K = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3,3)
    pt1 = np.random.randint(16,size=20).reshape(-1,2)
    pt2 = np.random.randint(16,size=20).reshape(-1,2)
    F,_ = cv2.findFundamentalMat(pt1,pt2,cv2.FM_8POINT)
    print(F)
    E = essentialMatrix(K,F)
    print(E)




