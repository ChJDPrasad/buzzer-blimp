from constants import *
import numpy as np


def calc_circumferential_stress(aerostat, v):
    envelope = aerostat.envelope
    alpha = aerostat.find_alpha(v)

    q = 0.5 * rho_air * v ** 2
    Pi = 1.15 * q

    Ni = (Pi + (rho_air - rho_gas) * g * envelope.c) * envelope.a
    Nb = np.pi * envelope.a * envelope.c * g * (rho_air - rho_gas) / 2
    Nbm = 0.123 * q * (4. * np.pi * envelope.a ** 3 / 3.) ** (1. / 3)
    Na = 0.05 * q * envelope.a * alpha

    return (Ni + Nb + Nbm + Na) * 50 / (9.81 * 10 ** 3)


def calc_tether_stress(aerostat, v):
    force = aerostat.calc_force(aerostat.find_alpha(v), v)

    return (force[0] ** 2 + force[1] ** 2) ** 0.5 / aerostat.tether.area