from __future__ import print_function
from constants import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, arctan
from numba import jit, njit
from scipy.optimize import fsolve
from scipy.misc import derivative


@njit
def sec(x):
    return 1. / np.cos(x)


class Kite(object):
    def __init__(self, lk, bk, hk,
                 spine_density=0.1, kite_density=rho_kite, rod_density=0.08):
        self.lk = lk
        self.bk = bk
        self.hk = hk
        self.spine_density = spine_density
        self.envelope_density = kite_density
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

    def print_info(self):
        pass

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

    @jit
    def calc_profile(self, Tx, Ty, cd=1., ds=0.2):
        n = int(self.length // ds + 2)
        theta0 = np.arctan(Ty / Tx)
        theta = np.zeros((n,))
        x, y = np.zeros_like(theta), np.zeros_like(theta)
        T = np.zeros_like(theta)
        radius = self.radius

        theta[0] = theta0

        T[0] = (Tx ** 2 + Ty ** 2) ** 0.5
        dW = self.density * ds * 9.81

        for i in range(n - 1):
            dD = 0.5 * 1.225 * (v * sin(theta[i])) ** 2 * (2 * radius) * ds * cd

            theta[i + 1] = arctan((T[i] * sin(theta[i]) - dW - dD * cos(theta[i])) /
                                  (T[i] * cos(theta[i]) + dD * sin(theta[i])))
            T[i + 1] = (T[i] * cos(theta[i]) + dD * sin(theta[i])) / cos(theta[i + 1])

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

    def print_info(self):
        pass

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
        return (-0.3269 * (aoa) ** 2 + 0.8036 * (aoa) + 0.0049)

    def calc_cd(self, aoa):
        return (0.3447 * (aoa) ** 2 + 0.0631 * (aoa) + 0.0989)

    def calc_cm(self, aoa):
        return -(0.2071 * (aoa) ** 2 - 0.5647 * (aoa) + 0.0012)

    def calc_force(self, aoa, v):
        net_force = np.zeros(2)
        q = 0.5 * rho_air * v ** 2
        net_force[0] = q * self.area * self.calc_cd(aoa)
        net_force[1] = q * self.area * self.calc_cl(aoa) - self.weight * g + self.buoyancy * g

    def print_info(self):
        print("Semi-major axis:\t%.3f" % self.a)
        print("Semi-minor axis:\t%.3f" % self.c)
        print("Envelope Surface Area:\t%.3f" % self.area)
        print("Envelope Volume:\t%.3f" % self.volume)
        print("Envelope Weight:\t%.3f" % self.weight)
        print("Buoyancy\t:%.3f" % self.buoyancy * g)


class Aerostat(object):
    def __init__(self, envelope, kite, tether, xc, zc, v=None):
        self.envelope = envelope
        self.kite = kite
        self.tether = tether
        self.xc = xc
        self.zc = zc
        self.v = v
        self.aoa = None
        if self.v is not None:
            self.aoa = self.find_alpha(self.v)

    def calc_force(self, aoa, v=None):
        v = self.v if v is None else v
        net_force = np.zeros(2)
        net_force += self.kite.calc_force(aoa, v)
        net_force += self.envelope.calc_force(aoa, v)
        return net_force

    def calc_moment(self, aoa, v):
        v = self.v if v is None else v
        net_moment = 0.
        net_moment += self.kite.calc_moment(aoa, v, -self.xc, -self.zc)
        net_moment += self.envelope.calc_moment(aoa, v, -self.xc, -self.zc)

        return net_moment

    def calc_moment_derivative(self, aoa, v):
        v = self.v if v is None else v
        return derivative(self.calc_moment, aoa, args=(v,))

    def find_alpha(self, v):
        v = self.v if v is None else v
        return fsolve(self.calc_moment, x0=0.1, args=(v,))[0]

    def set_velocity(self, v):
        self.v = v
        self.aoa = self.find_alpha(v)

    """
    Helper functions
    """

    @property
    def cm(self):
        assert (self.v is not None)
        q = 0.5 * rho_air * self.v ** 2
        return self.calc_moment(self.aoa, self.v) / (q * self.envelope.ref_area)

    @property
    def cm_alpha(self):
        assert (self.v is not None)
        q = 0.5 * rho_air * self.v ** 2
        return self.calc_moment_derivative(self.aoa, self.v) / (q * self.envelope.ref_area)

    @property
    def blowby(self):
        assert (self.v is not None)
        force = self.calc_force(self.aoa, self.v)
        return self.tether.calc_blowby(force[0], force[1])

    @property
    def altitude(self):
        assert (self.v is not None)
        force = self.calc_force(self.aoa, self.v)
        return self.tether.calc_altitude(force[0], force[1])

    def print_info(self):
        assert (self.v is not None)
        self.envelope.print_info()
        self.tether.print_info()
        self.kite.print_info()
        print("cm\t%.3f" % self.cm)
        print("cm_alpha:\t%.3f" % self.cm_alpha)
        print("Angle of Attack:\t%.3f" % self.aoa)
        print("Blowby:\t%.3f" % self.blowby)
        print("Altitude:\t%.3f" % self.altitude)
