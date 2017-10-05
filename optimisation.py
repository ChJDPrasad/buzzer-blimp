from __future__ import print_function
from analyzer import *
# from pso_generic import *
from pyswarm import pso

cost = lambda x: analyze(x[0], 1.2, x[1])[0]


def constraint_chk(x):
    ans = analyze(x[0], x[1])
    return (x[0] > 0 and x[0] < 15 and x[1] > 0 and x[1] < 15 and ans[4]["alpha"] > 0)

g, f = pso(cost, [0., 0.], [15., 15.])
print(g, f)
# swarm = Swarm(20, cost, dimension=2, constraint_chk=constraint_chk)
# print(swarm.optimise(50))
