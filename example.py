from filterlib import KalmanFilter, ExtendKalmanFilter
import numpy as np
import matplotlib.pyplot as plt

'''
1. Linear case: 

2. Nonlinear case:

3. BOT
'''
# Linear system
# measurement dim is 1
dt = 1.0 / 60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0])
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([1])
x0 = np.zeros(3)
samples_num = 100

def generate_data(F, H, Q, R, x0, samples_num):
    x = x0
    real_data, measurement_data = [], []
    for _ in range(samples_num):
        x = np.dot(F, x) + np.random.multivariate_normal([0, 0, 0], Q)
        z = np.dot(H, x) + np.random.normal(R)
        real_data.append(x)
        measurement_data.append(z)

    return np.array(real_data), np.array(measurement_data)




# Nonlinear system


# BOT system


if __name__ == '__main__':
    real, obs = generate_data(F, H, Q, R, x0, samples_num)
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R, x0 = x0)
    es = kf.run_kf(obs)
    print(real.shape, obs.shape, es.shape)

    plt.plot(real[:, 0], label = 'real')
    plt.plot(es[:, 0], label = 'es')
    plt.legend()
    plt.show()

