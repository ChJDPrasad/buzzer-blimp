import numpy as np
from numpy import sin, cos, tan
from scipy.optimize import fsolve
from constants import *
from scipy.misc import derivative
from numba import jit, njit
from common import Aerostat


def Mcp(alpha, kite, envelope, x, z, v):
    return Aerostat(envelope, kite, x, z).calc_moment(alpha, v)


def Mcp_alpha(alpha, kite, envelope, x, z, q):
    return derivative(Mcp, alpha, args=(kite, envelope, x, z, v))


