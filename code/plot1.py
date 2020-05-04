import numpy as np
import matplotlib.pyplot as plt




fig,ax = plt.subplots()
x,y = np.loadtxt('resultcv.csv', delimiter=',', unpack=True)
x2,y2 = np.loadtxt('result.csv', delimiter=',', unpack=True)

ax.plot(x,y,'o')
ax.plot(x2,y2,'o')

plt.show()
# for i in range(len(x)):
    # ax.plot(x[i],y[i],'o')
    # plt.pause(0.01)



