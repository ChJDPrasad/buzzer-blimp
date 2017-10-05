from constants import *
import numpy as np


def calc_circumferential_stress(v, alpha, envelope):
    q = 0.5 * rho_air * v ** 2
    Pi = 1.15 * q
    V = envelope.volume

    Ni = (Pi + (rho_air - rho_gas) * g * envelope.c) * envelope.a
    Nb = np.pi * envelope.a * envelope.c * g * (rho_air - rho_gas) / 2
    Nbm = 0.123 * q * (4. * np.pi * envelope.a ** 3 / 3.) ** (1. / 3)
    Na = 0.05 * q * envelope.a * alpha

    return (Ni + Nb + Nbm + Na) * 50 / (9.81 * 10 ** 3)


def calc_tether_stress(tether, Tx, Ty):
    return (Tx ** 2 + Ty ** 2) ** 0.5 / tether.area
