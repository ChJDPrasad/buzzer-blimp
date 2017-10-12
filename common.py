from constants import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, arctan

def sec(x):
    return 1. / np.cos(x)


class Kite(object):
    def __init__(self, lk, bk, hk,
                 spine_density=0.1, envelope_density=0.08258, rod_density=0.08):
        self.lk = lk
        self.bk = bk
        self.hk = hk
        self.spine_density = spine_density
        self.envelope_density = envelope_density
        self.rod_density = rod_density

    @property
    def weight(self):
        env_weight = (self.horizontal_area + self.vertical_area) * self.envelope_density
        struc_weight = self.spine_density * self.lk + self.rod_density * self.hk

        return env_weight + struc_weight

    @property
    def horizontal_area(self):
        return 0.5 * self.lk * self.bk

    @property
    def vertical_area(self):
        return 0.5 * self.lk * self.hk

    def calc_force(self, alpha, v):
        fz = 0.5 * rho_air * v**2 * self.horizontal_area * self.calc_cl(alpha) - self.weight * g
        fx = 0.5 * rho_air * v**2 * self.calc_cd_with_area(alpha)
        return np.array([fx, fz])

    def calc_moment(self, alpha, v, x, z):
        L = 0.5 * rho_air * v**2 * self.horizontal_area * self.calc_cl(alpha)
        D = 0.5 * rho_air * v**2 * self.calc_cd_with_area(alpha)
        M = 0.5 * rho_air * v**2 * self.horizontal_area * self.calc_cm(alpha)
        return (self.lk / 3 * cos(alpha) - x) * self.weight * g - (self.lk / 2 * cos(alpha) - x) * L +\
                M + D * (-lk / 2 * sin(alpha) - y)
    
    def calc_cl(self, alpha):
        return (0.9848 * (alpha) ** 2 + 0.7665 * (alpha) + 0.1002)

    def calc_cd_with_area(self, aoa):
        return (1.8524 * (alpha) ** 2 - 0.1797 * (alpha)) * self.horizontal_area + \
                0.1536 * (self.horizontal_area + self.vertical_area))

    def calc_cm(self, alpha):
        return -(0.2939 * (alpha) ** 2 + 0.5189 * (alpha) - 0.1921)


class Tether(object):
    def __init__(self, length, density=rho_tether, radius=r_tether):
        self.length = length
        self.density = density
        self.radius = radius

    @property
    def weight(self):
        return self.density * self.length

    @property
    def area(self):
        return np.pi * self.radius ** 2

    def _displacements(self, Tx, Ty):
        x, y = self.calc_profile(Tx, Ty)
        return x[-1], y[-1]

    def calc_profile(self, Tx, Ty, cd=1., ds=0.1):
        n = int(self.length // ds + 2)
        theta0 = np.arctan(Ty / Tx)

        theta = np.zeros((n,))
        T = np.zeros_like(theta)

        theta[0] = theta0

        T[0] = (Tx ** 2 + Ty ** 2) ** 0.5
        dW = self.density * ds * 9.81

        for i in range(n - 1):
            dD = 0.5 * 1.225 * (v * sin(theta[i])) ** 2 * (2 * self.radius) * ds * cd

            theta[i + 1] = arctan((T[i] * sin(theta[i]) - dW - dD * cos(theta[i])) /
                                  (T[i] * cos(theta[i]) + dD * sin(theta[i])))
            T[i + 1] = (T[i] * cos(theta[i]) + dD * sin(theta[i])) / cos(theta[i + 1])

        x, y = np.zeros_like(theta), np.zeros_like(theta)
        for i in range(1, n):
            x[i] = x[i - 1] + ds * cos(theta[n - i - 1])
            y[i] = y[i - 1] + ds * sin(theta[n - i - 1])
        return x, y

    def plot_profile(self, Tx, Ty):
        x, y = self.calc_profile(Tx, Ty)
        plt.plot(x, y)

    def calc_blowby(self, Tx, Ty):
        return self._displacements(Tx, Ty)[0]

    def calc_altitude(self, Tx, Ty):
        return self._displacements(Tx, Ty)[1]


class Envelope(object):
    def __init__(self, a, phi, density=rho_envelope):
        self.a = a
        self.phi = phi
        self.density = density

    @property
    def volume(self):
        a, c = self.a, self.a / self.phi
        return 4. * np.pi * c * a ** 2 / 3.

    @property
    def c(self):
        return self.a / self.phi

    @property
    def weight(self):
        return self.area * self.density

    @property
    def ref_area(self):
        return self.volume ** (2. / 3)

    @property
    def buoyancy(self):
        return self.volume * (rho_air - rho_gas)

    @property
    def area(self):
        a, c = self.a, self.a / self.phi
        e = 1. - c ** 2 / a ** 2
        return 2 * np.pi * a ** 2 * (1. + (1. - e ** 2) / e * np.arctanh(e))

    def calc_cl(self, aoa):
        raise NotImplementedError
        # return 0.2155 * aoa ** 3 - 0.468 * aoa ** 2 + 0.8261 * aoa + 0.0044

    def calc_cd(self, aoa):
        raise NotImplementedError
        # return -0.5924 * aoa ** 3 + 0.7324 * aoa ** 2 + 0.0013 * aoa + 0.1001
