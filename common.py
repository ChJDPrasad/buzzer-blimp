from constants import *
import numpy as np
import matplotlib.pyplot as plt


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

    def calc_cl(self, aoa):
        raise NotImplementedError

    def calc_cd(self, aoa):
        raise NotImplementedError


class Tether(object):
    def __init__(self, length, density=rho_tether):
        self.length = length
        self.density = density

    @property
    def weight(self):
        return self.density * self.length

    def _displacements(self, Tx, Ty):
        x, y, theta0, theta1 = self.calc_profile(Tx, Ty)
        return x(theta1), y(theta1)

    def calc_profile(self, Tx, Ty):
        K = (Tx / (self.density * 9.81))
        theta0 = np.arctan((Ty - self.density * 9.81 * self.length) / Tx)
        theta1 = np.arctan(Ty / Tx)
        x = lambda theta: K * (np.log(np.abs(sec(theta) + np.tan(theta))) - \
                               np.log(np.abs(sec(theta0) + np.tan(theta0))))
        y = lambda theta: K * (sec(theta) - sec(theta0))

        return x, y, theta0, theta1

    def plot_profile(self, Tx, Ty):
        x, y, theta0, theta1 = self.calc_profile(Tx, Ty)
        theta = np.linspace(theta0, theta1, 1000)
        x, y = x(theta), y(theta)
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