import numpy as np
import matplotlib.pyplot as plt
import cv2 


fig,ax = plt.subplots()
x,y = np.loadtxt('resultcv.csv', delimiter=',', unpack=True)
x2,y2 = np.loadtxt('result.csv', delimiter=',', unpack=True)


cap = cv2.VideoCapture('../input/inputVideo.avi')
i = 0
while(cap.isOpened()):
    _,frame = cap.read()
    ax.plot(x[i],-y[i],'bo')
    ax.plot(x2[i],y2[i],'ro')

    plt.pause(0.000001)
    cv2.imshow('img2',frame)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows() 



