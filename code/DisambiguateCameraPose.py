import cv2

def disambiguateCameraPose(Cset,Rset,Xset):
    bestC = None; bestR = None; bestCount = 0
    for i in range(4):
        count = 0
        num,_ = Xset[i].shape
        for j in range(num):
            z = Xset[i][j, 2] 
            cond = Rset[i][2,:].dot(Xset[i][j,:]-Cset[i]) 
            if(cond>0 and z>0):             
                count += 1
        if count > bestCount:
            bestC = Cset[i]
            bestR = Rset[i]
            bestCount = count

    if(bestC is not None):
           bestC = bestC.reshape(-1,1)
    return bestC, bestR 

if __name__ == '__main__':
    pass


