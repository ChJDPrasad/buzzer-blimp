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


def Mcp(alpha, z, q, W, B, Wk, S, Sk, a, b, cr):
    Lk = q * Sk * Clk(alpha)
    Dk = q * Sk * Cdk(alpha)
    Mk = q * Sk * cr * Cmk(alpha)
    L = q * S * Cl(alpha)
    D = q * S * Cd(alpha)
    M = q * S * 2 * a * Cm(alpha)

    return Wk * (z * sin(alpha) + cr / 3 * cos(alpha)) + (W - L - B) * (b + z) * \
                                                         sin(alpha) - Lk * (z * sin(alpha) + cr / 2 * cos(alpha)) + \
           Mk + M + D * (b + z) * cos(alpha) + Dk * (z * cos(alpha) - cr / 2 * sin(alpha))


def Mcp_alpha(alpha, z, q, W, B, Wk, S, Sk, a, b, cr):
    return derivative(Mcp, alpha, args=(z, q, W, B, Wk, S, Sk, a, b, cr))


def get_aerod_data(z, v, W, B, Wk, S, Sk, a, b, cr):
    q = 0.5 * rho_air * v ** 2
    alpha = fsolve(Mcp, 0.1, args=(z, q, W, B, Wk, S, Sk, a, b, cr))[0]

    Lk = q * Sk * Clk(alpha)
    Dk = q * Sk * Cdk(alpha)
    Mk = q * Sk * cr * Cmk(alpha)
    L = q * S * Cl(alpha)
    D = q * S * Cd(alpha)
    M = q * S * 2 * a * Cm(alpha)

    return {"alpha": alpha, "L": L, "Lk": Lk, "D": D, \
            "Dk": Dk, "M": M, "Mk": Mk}


def calc_Ty(alpha, v, W, Wk, B, S, Sk):
    q = 0.5 * rho_air * v ** 2
    Lk = q * Sk * Clk(alpha)
    L = q * S * Cl(alpha)
    return L + Lk + B - W - Wk


def calc_Tx(alpha, v, S, Sk):
    q = 0.5 * rho_air * v ** 2
    Dk = q * Sk * Cdk(alpha)
    D = q * S * Cd(alpha)
    return D + Dk
