from analyzer import *
from pso_generic import *

cost = lambda x: analyze(x[0], x[1])[0]
def constraint_chk(x):
    ans = analyze(x[0], x[1])
    return (x[0] > 0 and x[0] < 15 and x[1] > 0 and x[1] < 15 and ans[4]["alpha"] > 0)
swarm = Swarm(50, cost, dimension=2, constraint_chk=constraint_chk) 
swarm.optimise(50)
