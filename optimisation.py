from __future__ import print_function
from analyzer import *
from pyswarm import pso


def analyze_local(x):
    return analyze(x[0], x[1], x[2])


cost = lambda x: analyze_local(x)[0]


def stability_constraints(x, *args):
    r = analyze_local(x)
    return -r[4]['Cm_alpha'] - 2.


def stability_constraints2(x, *args):
    r = analyze_local(x)
    return r[4]['alpha']


g, f = pso(cost, [0., 0., 0.], [15., 5., 5.], ieqcons=[stability_constraints], debug=True)
print(g, f)
