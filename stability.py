import numpy as np
from numpy import sin, cos, tan
from scipy.optimize import fsolve
from constants import *
from scipy.misc import derivative


def Clk(alpha):
    return (0.9848 * (alpha) ** 2 + 0.7665 * (alpha) + 0.1002)


def Cdk(alpha):
    return (1.8524 * (alpha) ** 2 - 0.1797 * (alpha) + 0.1536)


def Cmk(alpha):
    return -(0.2939 * (alpha) ** 2 + 0.5189 * (alpha) - 0.1921)


def Cl(alpha):
    return (-0.3269 * (alpha) ** 2 + 0.8036 * (alpha) + 0.0049)


def Cd(alpha):
    return (0.3447 * (alpha) ** 2 + 0.0631 * (alpha) + 0.0989)


def Cm(alpha):
    return -(0.2071 * (alpha) ** 2 - 0.5647 * (alpha) + 0.0012)


def Mcp(alpha, kite, envelope, z, q):
    a, b = envelope.a, envelope.a / envelope.phi
    B = envelope.buoyancy * 9.8
    W = (envelope.weight + w_excess) * 9.8
    S = envelope.ref_area

    cr = kite.lk
    Wk = kite.weight * 9.8
    Sk = kite.horizontal_area

    Lk = q * Sk * Clk(alpha)
    Dk = q * Sk * Cdk(alpha)
    Mk = q * Sk * cr * Cmk(alpha)
    L = q * S * Cl(alpha)
    D = q * S * Cd(alpha)
    M = q * S * 2 * a * Cm(alpha)

    return Wk * (z * sin(alpha) + cr / 3 * cos(alpha)) + (W - L - B) * (b + z) * \
                                                         sin(alpha) - Lk * (z * sin(alpha) + cr / 2 * cos(alpha)) + \
           Mk + M + D * (b + z) * cos(alpha) + Dk * (z * cos(alpha) - cr / 2 * sin(alpha))


def get_aerod_data(z, v, kite, envelope):
    q = 0.5 * rho_air * v ** 2
    alpha = fsolve(Mcp, 0.1, args=(kite, envelope, z, q))[0]

    Lk = q * kite.horizontal_area * Clk(alpha)
    Dk = q * kite.horizontal_area * Cdk(alpha)
    Mk = q * kite.horizontal_area * kite.lk * Cmk(alpha)
    L = q * envelope.ref_area * Cl(alpha)
    D = q * envelope.ref_area * Cd(alpha)
    M = q * envelope.ref_area * 2 * envelope.a * Cm(alpha)

    return {"alpha": alpha, "L": L, "Lk": Lk, "D": D, \
            "Dk": Dk, "M": M, "Mk": Mk}


def calc_Ty(alpha, v, kite, envelope):
    q = 0.5 * rho_air * v ** 2
    Lk = q * kite.horizontal_area * Clk(alpha)
    L = q * envelope.ref_area * Cl(alpha)
    return L + Lk + envelope.buoyancy * 9.8 - envelope.weight * 9.8 - w_excess * 9.8 - kite.weight * 9.8


def calc_Tx(alpha, v, kite, envelope):
    q = 0.5 * rho_air * v ** 2
    Dk = q * kite.horizontal_area * Cdk(alpha)
    D = q * envelope.ref_area * Cd(alpha)

    return D + Dk
