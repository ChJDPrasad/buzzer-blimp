from __future__ import print_function
from analyzer import *
from pyswarm import pso
import pandas as pd
import os


def analyze_local(x, print_stuff=False):
    if ITERATION is 2:
        return analyze(x[0], x[1], x[2], x[3],  print_stuff=print_stuff)
    elif ITERATION is 3:
        return analyze(x[0], x[1], x[2], x[3], x[4], print_stuff=print_stuff)


_cache = {}


def cache(x):
    x = tuple(x)
    if x not in _cache:
        _cache[x] = analyze_local(x)
    return _cache[x]


def cost(x):
    aerostat = cache(x)[0]
    return aerostat.envelope.volume


def stability_constraints(x, *args):
    r = cache(x)[0]
    return -r.cm_alpha - 0.7


def stability_constraints2(x, *args):
    r = cache(x)[0]
    return r.aoa


def structural_constraints1(x, *args):
    r = cache(x)[1]
    return -r + 8.


def tether_constraint(x, *args):
    r = cache(x)[2]
    return -r + 75 * 1e6


def blowby_constraint(x):
    r = cache(x)[0]
    return -r.blowby + 20.


# Sanity constriants
def kite_sanity(x):
    """
    Ensure kite length is smaller than 2.5 times semi-major
    """
    aerostat = cache(x)[0]
    return 2.5 * aerostat.envelope.a - x[0]


def kite_sanity2(x):
    """
    Ensure kite wingspan is smaller than diameter of envelope
    """
    aerostat = cache(x)[0]
    return 2 * aerostat.envelope.a - x[4]


def confluence_sanity(x):
    """
    Dynamics of tether between envelope and confluence point
    will need to be evaluated if the confluence point is really far away.
    Therefore, limiting confluence point vertical distance
    """
    aerostat = cache(x)[0]
    return 2. * aerostat.envelope.c - x[2]


def optimize_iteration2():
    g, f = pso(cost, [0.1, 0., 0., 15], [9., 10., 7., 350],
               ieqcons=[stability_constraints, stability_constraints2,
                        structural_constraints1, tether_constraint,
                        blowby_constraint,
                        kite_sanity,
                        confluence_sanity],
               debug=True, maxiter=100, swarmsize=200)

    fname = 'iter2.csv'
    df = pd.DataFrame({
        'kite_length': [g[0]],
        'xc': [g[1]],
        'zc': [g[2]],
        'free_lift': [g[3]],
        'volume': [f],
    })

    if not os.path.isfile(fname):
        df.to_csv(fname, index=False)
    else:
        df.to_csv(fname, mode='a', header=False, index=False)

    analyze_local(g, print_stuff=True)


def optimize_iteration3():
    g, f = pso(cost, [0.1, 0., 0., 15], [8., 0.01, 7., 350],
               ieqcons=[stability_constraints, stability_constraints2,
                        structural_constraints1, tether_constraint,
                        blowby_constraint,
                        kite_sanity, kite_sanity2,
                        confluence_sanity],
               debug=True, maxiter=200, swarmsize=200)

    fname = 'iter3.csv'
    df = pd.DataFrame({
        'kite_length': [g[0]],
        'xc': [g[1]],
        'zc': [g[2]],
        'free_lift': [g[3]],
        'kite_wingspan': [g[4]],
        'volume': [f],
    })

    if not os.path.isfile(fname):
        df.to_csv(fname, index=False)
    else:
        df.to_csv(fname, mode='a', header=False, index=False)

    analyze_local(g, print_stuff=True)


if __name__ == "__main__":
    optimize_iteration2()
