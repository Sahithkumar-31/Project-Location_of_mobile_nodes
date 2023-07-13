import math
import random
from numpy.random import randint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

lstx = randint(1,100,[12])
lsty = randint(1,100,[12])
print(lstx,lsty)
x = np.array(lstx)
x = np.sort(x)
y = np.array(lsty)
y = np.sort(y)
radian = list()
distance = list()
angle = list()
plotx = list()
ploty = list()
arx = np.array([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])
ary = np.array([0,0,0,0,0,0,0])

for id in range(len(x)):
    r = math.sqrt(pow(x[id],2)+pow(y[id],2))
    distance.append(r+0.1)

for id in range(len(x)):
    r = math.atan2(y[id],x[id])
    radian.append(r+0.2)
    d = math.degrees(r)
    angle.append(d)


arz = np.array(distance)
arradian = np.array(radian)
angl = np.array(angle)
for id in range(len(x)):
    x1 = arz[id]*math.cos(arradian[id])
    y1 = arz[id]*math.sin(arradian[id])
    plotx.append((x1))
    ploty.append((y1))

px = np.array(plotx)
py = np.array(ploty)



print(px,py)
print(angle)
plt.subplot(311)
plt.scatter(x,y,c = 'red')
plt.ylabel('Y axis')  
plt.xlabel('X axis')  
plt.scatter(arx,ary,c = 'blue')
plt.scatter(px,py,c = 'green')
plt.show()
plt.subplot(312)

plt.title('Error Detection Graph')
plt.plot(x,y)
plt.plot(px,py)
plt.scatter(px,py,marker = 'x',c = 'green')
plt.ylabel('Y axis')  
plt.xlabel('X axis')  
plt.show()

for id in range(len(x)):
  np.random.seed(0)

  doa = np.array([angl[id]])  # Direction of arrival
  N = 200  # Snapshots
  w = np.array([np.pi / 4])  # Frequency
  M = 10  # Number of array elements
  P = len(w)  # The number of signal
  lambd = 150  # Wavelength
  d = lambd / 2  # Element spacing
  snr = 20  # SNA

  D = np.zeros((M, P), dtype=np.complex128)  # To create a matrix with P row and M column
  for k in range(P):
    D[:, k] = np.exp(-1j * 2 * np.pi * d * np.sin(doa[k]) / lambd * np.arange(M))

  xx = 2 * np.exp(1j * np.outer(w, np.arange(N)))  # Simulate signal
  x = D @ xx
  x += np.random.randn(*x.shape) * np.sqrt(0.5 * 10 ** (-snr / 10))  # Insert Gaussian white noise

  R = x @ x.T.conj() / N  # Data covariance matrix
  N_, V = np.linalg.eig(R)  # Find the eigenvalues and eigenvectors of R
  NN = V[:, :M - P]  # Estimate noise subspace

  theta = np.arange(0, 90.5, 0.2)  # Peak search
  Pmusic = np.zeros_like(theta)
  for ii in range(len(theta)):
    SS = np.exp(-1j * 2 * np.pi * d * np.sin(theta[ii] / 180 * np.pi) / lambd * np.arange(M))
    PP = abs(SS @ NN @ NN.conj().T @ SS.conj())
    Pmusic[ii] = 1 / PP

  Pmusic = 10 * np.log10(Pmusic / Pmusic.max())  # Spatial spectrum function
  plt.subplot(313)
  plt.plot(theta, Pmusic, '-k')
  plt.xlabel('angle \u03b8/degree')
  plt.ylabel('spectrum function P(\u03b8) /dB')
  plt.title('DOA estimation based on MUSIC algorithm')
  plt.grid(True)
  plt.show()