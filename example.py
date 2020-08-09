from filterlib import KalmanFilter, ExtendKalmanFilter, IteratedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt

'''
we will test one-dimensional cases and a multidimensional cases for the following case
one-dimensional case:
cubic sensor : x_k = f(x_{k-1}) +v_k, y_k = h(x_k) + w_k
where f = x + x*cos(x), h = x^3

multidimensional case:
BOT : (x, y, v_x, v_y) where x, y, v_x, v_y are displacement at x-axis, displacement at y-axis
velocity at x-axis, velocity at y-axis
 
Tracking dynamics : (x, v, a) where x, v, a are displacement, velocity, acceleration

multidimensional cubic sensor
'''
default_param = {'f': lambda x: x + 0.1* x * np.cos(x), 'h': lambda x: np.power(x, 3),
                 'fg': lambda x: 1 + 0.1*(np.cos(x) - x*np.sin(x)), 'hg': lambda x: 3*np.power(x, 2)}

default_param_bot = {'f': lambda x: np.dot(np.array([[1, 0, 0.1, 0],
                           [0, 1, 0, 0.1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]), x),
                     'h': lambda x: np.array([np.sqrt(x[0]**2 + x[1]**2),
                                     np.arctan((x[1])/x[0])]),
                     'fg': None,
                     'hg': lambda x: np.array([[x[0]/(np.sqrt(x[0]**2 + x[1]**2)),
                                                x[1]/(np.sqrt(x[0]**2 + x[1]**2)), 0, 0],
                                               [-x[1]/(x[0]**2 + x[1]**2), x[0]/(x[0]**2 + x[1]**2), 0, 0]]),
                     'Q': np.array([[np.power(0.1, 3)/3, 0, np.power(0.1, 2)/2, 0],
                           [0, np.power(0.1, 3)/3, 0, np.power(0.1, 2)/2],
                           [np.power(0.1, 2)/2, 0, 0.1, 0],
                           [0, np.power(0.1, 2)/2, 0, 0.1]]),
                     'R': np.array([[((3*np.pi)/180)**2, 0.0],
                           [0.0,  0.1**2]]),
                     'F': np.array([[1, 0, 0.1, 0],
                           [0, 1, 0, 0.1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]),
                     'x0': np.array([50, 50, 0, 0])}

def plot_test(real_data, estimate_data, i):
    plt.plot(real_data[:, i], label = 'real_data')
    plt.plot(estimate_data[:, i], label = 'estimate_data')
    plt.legend()
    plt.show()

class CubicSensor(object):
    def __init__(self, m = None, n = None, f = None, h = None, Q = None, R = None,
                 N = None, x0 = None):
        if (m  == None or n == None):
            raise ('Please input system dimension n and measurement dimension m!')

        if (f , h, Q, R, N, x0 == [None]*6):
            print('Default parameters are as following: \n'
                  'f, h = x + x*cos(x), x^3 \n'
                  'Q, R, N, x0 = I, I, 50, normal(0, 1, n)')

        self.n = n
        self.m = m
        self.f = lambda x: x + 0.1*x * np.cos(x) if f is None else f
        self.h = lambda x: np.power(x, 3) if h is None else h
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.N = 50 if N is None else N
        self.x = np.random.normal(size = self.n) if x0 is None else x0
        self.mean_f = np.zeros(self.n)
        self.mean_h = np.zeros(self.m)

    def generate_data(self):
        real_data, measurement_data = [], []
        for _ in range(self.N):
            real_data.append(self.x)
            z = self.h(self.x) + np.random.multivariate_normal(self.mean_h, self.R)
            measurement_data.append(z)
            self.x = self.f(self.x) + np.random.multivariate_normal(self.mean_f, self.Q)

        return np.array(real_data), np.array(measurement_data)


class BearOnlyTrack(object):
    def __init__(self, m = None, n = None, f = None, h = None, Q = None, R = None,
                 N = None, x0 = None, T = None):

        self.n = 4 if n is None else n
        self.m = 2 if m is None else m
        self.T = 0.1 if T is None else T

        self.F = np.array([[1, 0, self.T, 0],
                           [0, 1, 0, self.T],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.f = lambda x: np.dot(self.F, x) if f is None else f

        self.h = lambda x: np.array([np.sqrt(x[0]**2 + x[1]**2),
                                     np.arctan((x[1])/x[0])]) if h is None else h

        q = np.array([[np.power(self.T, 3)/3, 0, np.power(self.T, 2)/2, 0],
                           [0, np.power(self.T, 3)/3, 0, np.power(self.T, 2)/2],
                           [np.power(self.T, 2)/2, 0, self.T, 0],
                           [0, np.power(self.T, 2)/2, 0, self.T]])

        self.Q = q if Q is None else Q

        r = np.array([[((3*np.pi)/180)**2, 0.0],
                           [0.0,  0.1**2]])

        self.R = r if R is None else R
        self.N = 500 if N is None else N
        self.x = np.array([50, 50, 0, 0]) if x0 is None else x0
        self.mean_f = np.zeros(self.n)
        self.mean_h = np.zeros(self.m)

    def generate_data(self):
        real_data, measurement_data = [], []
        for _ in range(self.N):
            real_data.append(self.x)
            z = self.h(self.x) + np.random.multivariate_normal(self.mean_h, self.R)
            measurement_data.append(z)
            self.x = self.f(self.x) + np.random.multivariate_normal(self.mean_f, self.Q)

        return np.array(real_data), np.array(measurement_data)

class TrackingDynamics(object):
    pass

if __name__ == '__main__':

    # CubicSensor test
    CS = CubicSensor(n = 2, m = 2)
    real_data, measurement_data = CS.generate_data()
    print(real_data.shape, measurement_data.shape)
    ekf = ExtendKalmanFilter(n = 2, m = 2, f = default_param['f'], h = default_param['h'],
                             fg = default_param['fg'], hg = default_param['hg'])

    iekf = IteratedKalmanFilter(n = 2, m = 2, f = default_param['f'], h = default_param['h'],
                             fg = default_param['fg'], hg = default_param['hg'])

    estimate_data_ekf = ekf.run_ekf(measurement_data)
    estimate_data_iekf = iekf.run_iekf(measurement_data)
    plot_test(real_data, estimate_data_ekf, 0)
    plot_test(real_data, estimate_data_iekf, 0)


    BOT = BearOnlyTrack()
    real_data_bot, measurement_data_bot = BOT.generate_data()
    #print(real_data_bot.shape, measurement_data_bot.shape)
    ekf_bot = ExtendKalmanFilter(n = 4, m = 2, f = default_param_bot['f'], h = default_param_bot['h'],
                                 fg = default_param_bot['fg'], hg = default_param_bot['hg'],
                                 Q = default_param_bot['Q'], R = default_param_bot['R'],
                                 F = default_param_bot['F'], x0 = default_param_bot['x0'])

    estimate_data_bot = ekf_bot.run_ekf(measurement_data_bot)
    #print(estimate_data_bot.shape)
    x_bot_est, y_bot_est = estimate_data_bot[:, 0], estimate_data_bot[:, 1]
    x_bot_real, y_bot_real = real_data_bot[:, 0], real_data_bot[:, 1]
    plt.plot(x_bot_real, y_bot_real, label = 'real_data')
    #plt.plot(x_bot_est, y_bot_est,  label = 'estimate_data')
    plt.legend()
    plt.show()
