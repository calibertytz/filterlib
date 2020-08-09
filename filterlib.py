import numpy as np
from scipy.stats import norm

class KalmanFilter(object):
    def __init__(self, n = None, m = None, F = None, H = None, Q = None, R = None,
                 P = None, x0 = None):

        if (m  == None or n == None):
            raise ("Please input system dimension n and measurement dimension m!")
        elif (F == None or H == None):
            raise ValueError("Set proper system dynamics.")

        self.n = n
        self.m = m
        self.F = F
        self.H = H

        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.random.normal(size = self.n) if x0 is None else x0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) if S.ndim > 1 \
            else np.dot(self.P, self.H.T)/S
        self.x = self.x + (np.dot(K, y)).T if y.ndim > 1 else self.x + K*y
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

    def run_kf(self, measurement_data):
        estimate_data = []
        for z in measurement_data:
            self.predict()
            estimate_data.append(self.update(z))
        return np.array(estimate_data)



class ExtendKalmanFilter(object):
    def __init__(self, n = None, m = None, f = None, h = None, fg = None, hg = None,
                 Q = None, R = None, P = None, x0 = None, F = None, H = None):

        if (m  == None or n == None):
            raise ("Please input system dimension n and measurement dimension m!")
        elif (f == None or h == None):
            raise ValueError("Set proper system dynamics.")

        self.n = n
        self.m = m
        self.f = f
        self.h = h
        self.fg = fg
        self.hg = hg
        self.F  = F
        self.H = H

        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.random.normal(size=self.n) if x0 is None else x0

    def predict(self):
        F = self.F if self.fg is None else self.fg(self.x)
        self.x = self.f(self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q
        return self.x

    def update(self, z):
        H = self.H if self.hg is None else self.hg(self.x)
        y = z - np.dot(H, self.x)
        S = self.R + np.dot(H, np.dot(self.P, H.T))
        K = np.dot(np.dot(self.P, H.T), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, H), self.P)
        return self.x

    def run_ekf(self, measurement_data):
        estimate_data = []
        for z in measurement_data:
            self.predict()
            estimate_data.append(self.update(z))
        return np.array(estimate_data)


class IteratedKalmanFilter(object):
    def __init__(self, n = None, m = None, f = None, h = None, fg = None, hg = None,
                 Q = None, R = None, P = None, x0 = None, itr = None):

        if (m  == None or n == None):
            raise ("Please input system dimension n and measurement dimension m!")
        elif (f == None or h == None or fg ==None or hg ==None):
            raise ValueError("Set proper system dynamics.")

        self.n = m
        self.m = n
        self.f = f
        self.h = h
        self.fg = fg
        self.hg = hg

        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.random.normal(size = self.n) if x0 is None else x0
        self.itr = 5 if itr is None else itr

    def predict(self):
        F = self.fg(self.x)
        self.x = self.f(self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z):
        x_hat = self.x
        for _ in range(self.itr):
            H = self.hg(self.x)
            S = self.R + np.dot(H, np.dot(self.P, H.T))
            K = np.dot(np.dot(self.P, H.T), np.linalg.pinv(S))
            W = np.dot(K,(z - self.h(self.x) - np.dot(H, x_hat - self.x)))
            self.x = x_hat + W
            I = np.eye(self.n)
            self.P = np.dot(I - np.dot(K, H), self.P)

        return self.x

    def run_iekf(self, measurement_data):
        estimate_data = []
        for z in measurement_data:
            self.predict()
            estimate_data.append(self.update(z))
        return np.array(estimate_data)


class UnscentedKalmanFilter(object):
    pass

class SIR(object):
    def __init__(self, n = None, m = None, f = None, h = None, Q = None, R = None, particle_num = None,
                 x_P = None, V = None):

        if (m  == None or n == None):
            raise ("Please input system dimension n and measurement dimension m!")
        elif (f is None or h is None):
            raise ValueError("Set proper system dynamics.")

        self.n = n
        self.m = m
        self.f = f
        self.h = h

        self.particle_num = 100 if particle_num is None else particle_num
        self.V = 1 if V is None else V
        self.x_P = np.random.normal(0, V, particle_num) if x_P is None else x_P
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""
        mean = np.average(particles, weights=weights)
        var = np.average((particles - mean) ** 2, weights=weights)
        return mean, var

    def simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        rn = np.random.rand(N)
        indexes = np.searchsorted(cumulative_sum, rn)
        # sampling by index
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)
        return particles, weights

    def run_sir(self, measurement_data):
        x_est_out = []
        for z in measurement_data:
            #sample from p(x(k) | x(k - 1))
            x_P_update = self.f(self.x_P) + np.random.normal(0, self.Q, self.particle_num)
            z_update = h(x_P_update)
            # compute particle weight
            P_w = norm.pdf(z - z_update, self.R)
            P_w /= np.sum(P_w)
            # resampling
            x_P, P_w = self.simple_resample(x_P_update, P_w)
            # estimate
            self.x, _ = self.estimate(x_P_update, P_w)

            x_est_out.append(self.x)

        return  x_est_out



class GaussianParticleFilter(object):
    def __init__(self, f = None, h = None, Q = None, R = None, particle_num = None,
                 mean = None, var = None):

        self.n = Q.shape[1]
        self.m = R.shape[1]
        self.particle_num = 100 if particle_num is None else particle_num

        self.f = f
        self.h = h
        self.mean = np.zeros(self.n) if mean is None else mean
        self.var = np.eye(self.n) if var is None else var
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R

    def run_gpf(self, measurement_data):
        x_est_p = []
        x_est_e = []
        for z in measurement_data:

            x_P = np.random.normal(self.mean, np.sqrt(self.var), self.particle_num)

            P_w = norm.pdf(z - z_update, self.R)
            P_w /= np.sum(P_w)

            mu, covar = estimate(x_P, P_w)

            sample1 = np.random.normal(mu, np.sqrt(covar), self.particle_num)
            sample2 = np.random.normal(self.f(sample1), self.Q)

            self.mean = sample2.mean()
            self.var = (sample2.std()) ** 2

            x_est_p.append(mu)
            x_est_e.append(self.mean)

        return  x_est_p, x_est_e
