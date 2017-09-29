import matplotlib.pyplot as plt
from stability import Clk, Cdk, Cmk
import numpy as np
import seaborn
from common import Tether

alpha = np.linspace(0., 0.5, 5)
plt.title('Cl vs alpha (rad)')
plt.plot(alpha, Clk(alpha), marker='o')
plt.savefig('plots/kite-cl.png', bbox_inches="tight")
plt.clf()

plt.title('Cd vs alpha (rad)')
plt.plot(alpha, Cdk(alpha), marker='o')
plt.savefig('plots/kite-cd.png', bbox_inches="tight")
plt.clf()

plt.title('Cm vs alpha (rad)')
plt.plot(alpha, Cmk(alpha), marker='o')
plt.savefig('plots/kite-cm.png', bbox_inches="tight")
plt.clf()

tether = Tether(100., 1.)
plt.title('Tether profile')
tether.plot_profile(100., 1081)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('plots/tether-profile.png', bbox_inches="tight")
plt.clf()
