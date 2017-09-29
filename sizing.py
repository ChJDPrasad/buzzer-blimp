import numpy as np
from constants import *
from scipy.optimize import newton

"""
Only does static sizing
"""


def calc_volume(a, c):
    return 4. * np.pi * c * a ** 2 / 3.


def calc_area(a, c):
    e = 1. - c ** 2 / a ** 2
    return 2 * np.pi * a ** 2 * (1. + (1. - e ** 2) / e * np.arctanh(e))


def find_a(phi, kite, tether):
    surface_area = lambda a: calc_area(a, a / phi)
    w_e = lambda a: rho_envelope * surface_area(a)
    lift = lambda a: (rho_air - rho_gas) * calc_volume(a, a / phi)

    # Excluding common gravity term
    w_total = lambda a: w_excess + tether.weight + w_e(a) + kite.weight
    free_lift = lambda a: (fl / 100.) * w_total(a)
    excess_payload = lambda a: lift(a) - w_total(a) - free_lift(a)

    a = newton(lambda a: excess_payload(a), 5.)
    return a
