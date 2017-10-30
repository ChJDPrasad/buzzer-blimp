import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from numba import jit


@jit
def find_A(c, n=11):
    theta = np.linspace(1e-7, pi - 1e-7, n)
    chord = np.empty(n)

    A = np.empty((n, n))
    for i in range(n):
        chord[i] = c(theta[i])

    for i in range(n):
        for j in range(n):
            A[i, j] = 2 * sin((j + 1) * theta[i]) / (pi * chord[i]) + (j + 1) * sin((j + 1) * theta[i]) / sin(theta[i])
    return A


def calc_fourier_coefficients(A, aoa=0.):
    """
    cos(theta) = -2y/b
    :param A: Matrix ...
    :param aoa: Angle of attack
    :return:
    """
    b = np.empty(A.shape[0])
    b[:] = aoa
    return np.linalg.solve(A, b)


def circulation(a, theta):
    res = 0.
    for i in range(len(a)):
        res += 2 * a[i + 1] * sin((i + 1) * theta)
    return res


def CL(a, aspect_ratio):
    return a[0] * pi * aspect_ratio


def CD(a, aspect_ratio):
    res = 0.
    for i in range(len(a)):
        res += pi * aspect_ratio * ((i + 1) * a[i] ** 2)
    return res


A_elliptical = {}


def calc_elliptical_props(aoa, aspect_ratio, n=11):
    global A_elliptical

    c_r = 4. / (pi * aspect_ratio)
    c_elliptical = lambda theta: c_r * (1. - cos(theta) ** 2) ** 0.5

    if (aspect_ratio, n) not in A_elliptical:
        A = find_A(c_elliptical, n)
        A_elliptical[(aspect_ratio, n)] = A
    else:
        A = A_elliptical[(aspect_ratio, n)]

    a = calc_fourier_coefficients(A, aoa)
    return CL(a, aspect_ratio), CD(a, aspect_ratio)


A_delta = {}


def calc_delta_props(aoa, aspect_ratio, n=11):
    global A_delta

    c_r = 2 / aspect_ratio
    c_delta = lambda theta: c_r * (1 - np.abs(cos(theta)))
    if (aspect_ratio, n) not in A_delta:
        A = find_A(c_delta, n)
        A_delta[(aspect_ratio, n)] = A
    else:
        A = A_delta[(aspect_ratio, n)]

    a = calc_fourier_coefficients(A, aoa)
    return CL(a, aspect_ratio), CD(a, aspect_ratio)


def fit_quad(aspect_ratio, props_func, n=11, n_fit=11):
    aoa = np.linspace(-0.5 * pi, 0.5 * pi, n_fit)
    cl, cd = zip(*[props_func(aoa=alpha, aspect_ratio=aspect_ratio, n=n)
                   for alpha in aoa])

    coeff_cl = np.polyfit(aoa, cl, deg=2)
    coeff_cd = np.polyfit(aoa, cd, deg=2)

    return coeff_cl, coeff_cd


_delta_cl_cd = {}


def delta_cl_cd(aspect_ratio):
    if aspect_ratio not in _delta_cl_cd:
        calc_cl, calc_cd = fit_quad(aspect_ratio, calc_delta_props)
        _delta_cl_cd[aspect_ratio] = (calc_cl, calc_cd)

    return (lambda alpha: np.polyval(_delta_cl_cd[aspect_ratio][0], alpha),
            lambda alpha: np.polyval(_delta_cl_cd[aspect_ratio][1], alpha))


if __name__ == "__main__":
    aoa = np.linspace(1e-7, pi / 6., 10)
    plt.plot(aoa, 2 * pi * aoa)
    plt.plot(aoa, [calc_elliptical_props(alpha, 2.)[0] for alpha in aoa])
    plt.plot(aoa, [calc_delta_props(alpha, 2.)[0] for alpha in aoa])
    plt.show()
    # aspect_ratio = 3.
    #
    # def find_e(aspect_ratio):
    #     a = np.array([calc_delta_props(alpha, aspect_ratio)[0] / alpha for alpha in aoa])
    #     a0 = np.array([2 * pi for alpha in aoa])
    #     e = np.average(a0 / (pi * aspect_ratio * (a0 / a - 1.)))
    #     return e
    #
    #
    # aspect_ratio = np.linspace(0.5, 10., 20.)
    # e = [find_e(_) for _ in aspect_ratio]
    # plt.plot(aspect_ratio, e)
    # print(e)
    aspect_ratio = 2.
    # fit_quad(2., calc_delta_props)
    plt.plot(aoa, [calc_elliptical_props(alpha, aspect_ratio)[1] for alpha in aoa], marker='o')
    # plt.plot(aoa, [calc_delta_props(alpha, aspect_ratio)[1] for alpha in aoa])
    plt.plot(aoa, [calc_elliptical_props(alpha, aspect_ratio)[0] ** 2 / (pi * aspect_ratio) for alpha in aoa], '--')
    plt.show()
