from __future__ import print_function

from common import Tether, Kite, Envelope, Aerostat
from stability import *
from sizing import find_a
from constants import *
import numpy as np
import matplotlib.pyplot as plt
from structures import calc_circumferential_stress, calc_tether_stress


def print_analysis(aerod_data, Tx, Ty, blowby, altitude, N_circ, s_tether):
    print('aoa: ' + str(aerod_data["alpha"]))
    print("Tx: ", Tx)
    print("Ty: ", Ty)
    print("Cm_alpha: ", aerod_data["Cm_alpha"])
    print("Blowby: ", blowby)
    print("Altitude: ", altitude)
    print("Envelope circumferential stress: ", N_circ)
    print("Tether stress: ", s_tether)


def print_info(tether, kite, envelope):
    total_weight = envelope.weight + kite.weight + tether.weight + w_excess

    print("Semi-major axis", envelope.a)
    print("Semi-minor axis", envelope.c)

    print("Kite wingspan", kite.bk)
    print("Kite root chord", kite.lk)

    print("Envelope volume", envelope.volume)
    print("Buoyancy", envelope.buoyancy)
    print("Tether weight", tether.weight)
    print("Envelope weight", envelope.weight)
    print("Kite weight", kite.weight)

    print("Total weight", total_weight)


def sizing(lk, x, z, free_lift, bk):
    if bk is None:
        bk = 0.54 * lk

    # kite = Kite(lk, 0.5 * 1.08 * lk, 0.)
    kite = Kite(lk, bk, 0.25 * bk)
    tether = Tether(100.)

    a = find_a(1.6, kite, tether, free_lift)
    envelope = Envelope(a, 1.6)

    return Aerostat(envelope, kite, tether, x, z, v)


def analyze(lk=4.5, x=0., z=3., free_lift=70, bk=None, print_stuff=False):
    aerostat = sizing(lk, x, z, free_lift, bk)
    N_circ = calc_circumferential_stress(aerostat, v)
    s_tether = calc_tether_stress(aerostat, v)

    if print_stuff:
        aerostat.print_info()
        print("Circumferential stress: %.3f" % N_circ)
        print("Tether stress: %.3f MPa" % (s_tether / 1e6))
    return aerostat, N_circ, s_tether


if __name__ == "__main__":
    # analyze(0.94746596, 14.996898, 1.05079009 print_stuff=True)
    analyze(0.1, 0.0516, 4.99, 70, 1.2, print_stuff=True)
