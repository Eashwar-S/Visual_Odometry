import numpy as np
import matplotlib.pyplot as plt




fig,ax = plt.subplots()
x,y = np.loadtxt('resultcv.csv', delimiter=',', unpack=True)
x,y = np.loadtxt('result.csv', delimiter=',', unpack=True)

for i in range(len(x)):
    ax.plot(x[i],y[i],'o')
    plt.pause(0.01)



