from __future__ import print_function
from analyzer import *
from pyswarm import pso


def analyze_local(x, print_stuff=False):
    return analyze(x[0], x[1], x[2], x[3], print_stuff=print_stuff)


def cost(x):
    aerostat = analyze_local(x)[0]
    return aerostat.envelope.volume


# cost = lambda x: analyze_local(x)[0].envelope.volume + analyze_local(x)[0].env


def stability_constraints(x, *args):
    r = analyze_local(x)[0]
    return -r.cm_alpha - 0.7


def stability_constraints2(x, *args):
    r = analyze_local(x)[0]
    return r.aoa


def structural_constraints1(x, *args):
    r = analyze_local(x)[1]
    return -r + 8.


def tether_constraint(x, *args):
    r = analyze_local(x)[2]
    return -r + 75 * 1e6


def blowby_constraint(x):
    r = analyze_local(x)[0]
    return -r.blowby + 25.


# Sanity constriants
def kite_sanity(x):
    """
    Ensure kite length is smaller than diameter of envelope
    """
    aerostat = analyze_local(x)[0]
    return 2 * aerostat.envelope.a - x[0]

g, f = pso(cost, [0.1, 0., 0., 15], [15., 5., 5., 500],
           ieqcons=[stability_constraints, stability_constraints2,
                    structural_constraints1, tether_constraint,
                    blowby_constraint, kite_sanity],
           debug=True, maxiter=20)
analyze_local(g, print_stuff=True)
