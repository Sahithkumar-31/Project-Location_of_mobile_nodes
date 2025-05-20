import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

#Generating the Data
lstx = randint(1, 100, [12]) #X co-ordinates of the solder
lsty = randint(1, 100, [12]) #Y co-ordinates of the soldier
x = np.sort(np.array(lstx))
y = np.sort(np.array(lsty))

radian = []
distance = []
angle = []
plotx = []
ploty = []

#Sensor co-ordinates
arx = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
ary = np.array([0, 0, 0, 0, 0, 0, 0])

for id in range(len(x)):
    r = math.sqrt(x[id]**2 + y[id]**2)
    distance.append(r + 0.1)

for id in range(len(x)):
    r = math.atan2(y[id], x[id])
    radian.append(r + 0.2)
    angle.append(math.degrees(r))

arz = np.array(distance)
arradian = np.array(radian)
angl = np.array(angle)

for id in range(len(x)):
    x1 = arz[id] * math.cos(arradian[id])
    y1 = arz[id] * math.sin(arradian[id])
    plotx.append(x1)
    ploty.append(y1)

px = np.array(plotx)
py = np.array(ploty)

plt.subplot(311)
plt.scatter(x, y, c='red', label='Original')
plt.scatter(arx, ary, c='blue', label='Array')
plt.scatter(px, py, c='green', label='MUSIC Output')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.title('Node Positions')
plt.grid(True)

plt.subplot(312)
plt.title('Error Detection Graph')
plt.plot(x, y, label='Original')
plt.plot(px, py, label='MUSIC Output')
plt.scatter(px, py, marker='x', c='green')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)

#MUSIC algo implementation for every co-ordinate
for id in range(len(x)):
    np.random.seed(0)
    doa = np.array([angl[id]])
    N = 200
    w = np.array([np.pi / 4])
    M = 10
    P = len(w)
    lambd = 150
    d = lambd / 2
    snr = 20

    D = np.zeros((M, P), dtype=np.complex128)
    for k in range(P):
        D[:, k] = np.exp(-1j * 2 * np.pi * d * np.sin(doa[k]) / lambd * np.arange(M))

    xx = 2 * np.exp(1j * np.outer(w, np.arange(N)))
    x_signal = D @ xx
    x_signal += np.random.randn(*x_signal.shape) * np.sqrt(0.5 * 10 ** (-snr / 10))

    R = x_signal @ x_signal.T.conj() / N
    N_, V = np.linalg.eig(R)
    NN = V[:, :M - P]

    theta = np.arange(0, 90.5, 0.2)
    Pmusic = np.zeros_like(theta)
    for ii in range(len(theta)):
        SS = np.exp(-1j * 2 * np.pi * d * np.sin(theta[ii] / 180 * np.pi) / lambd * np.arange(M))
        PP = abs(SS @ NN @ NN.conj().T @ SS.conj())
        Pmusic[ii] = 1 / PP

    Pmusic = 10 * np.log10(Pmusic / Pmusic.max())
    plt.subplot(313)
    plt.plot(theta, Pmusic, '-k')
    plt.xlabel('angle \u03b8 (degrees)')
    plt.ylabel('P(\u03b8) / dB')
    plt.title('MUSIC Spectrum for AoA Estimation')
    plt.grid(True)

plt.tight_layout()
plt.show()

#Prediction
# Combine (x, y) as time series
data = np.stack((x, y), axis=1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, look_back=3):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

look_back = 3
X_ts, y_ts = create_dataset(scaled_data, look_back)
X_ts = X_ts.reshape((X_ts.shape[0], look_back, 2))

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 2)))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')
model.fit(X_ts, y_ts, epochs=200, verbose=0)

predicted = []
input_seq = X_ts[-1]
for _ in range(5):
    pred = model.predict(input_seq.reshape(1, look_back, 2), verbose=0)
    predicted.append(pred[0])
    input_seq = np.append(input_seq[1:], [pred[0]], axis=0)

predicted_positions = scaler.inverse_transform(predicted)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Original Trajectory', marker='o')
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted Trajectory (LSTM)', marker='x')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('LSTM Forecast of Mobile Node Trajectory')
plt.legend()
plt.grid(True)
plt.show()
