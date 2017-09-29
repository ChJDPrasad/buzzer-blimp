from common import Tether, Kite, Envelope
from stability import *
from sizing import find_a
from constants import *
import numpy as np
import matplotlib.pyplot as plt


def print_info(tether, kite, envelope):
    total_weight = envelope.weight + kite.weight + tether.weight + w_excess

    print("Semi-major axis", envelope.a)
    print("Semi-minor axis", envelope.c)

    print("Buoyancy", envelope.buoyancy)
    print("Tether weight", tether.weight)
    print("Envelope weight", envelope.weight)
    print("Kite weight", kite.weight)

    print("Total weight", total_weight)

def analyze(lk=5, z=1.6):
    kite = Kite(lk, 0.5 * 1.08 * lk, 0.)
    tether = Tether(100.)

    a = find_a(1.6, kite, tether)
    envelope = Envelope(a, 1.6)
    sph_payload_weight = envelope.weight + w_excess

    aoa = get_aerod_data(z,v, (sph_payload_weight)*g,envelope.buoyancy*g, kite.weight * g,
                    envelope.ref_area, kite.horizontal_area,
                    envelope.a, envelope.c, lk)["alpha"]

    Ty = calc_Ty(aoa, v, sph_payload_weight * g, kite.weight * g,
                envelope.buoyancy * g, envelope.ref_area, kite.horizontal_area)
    Tx = calc_Tx(aoa, v, envelope.ref_area, kite.horizontal_area)

    print('aoa: ' + str(aoa))
    print("Tx: ", Tx)
    print("Ty: ", Ty)   
    print("Blowby: ", tether.calc_blowby(Tx, Ty))
    print("Altitude: ", tether.calc_altitude(Tx, Ty))

if __name__ == "__main__":
    analyze(6, 2)
