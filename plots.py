import matplotlib.pyplot as plt
from stability import Clk, Cdk, Cmk
import numpy as np
import seaborn
from common import Tether
from analyzer import analyze

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

plt.title("Blowby variation vs Kite length")
kl = np.linspace(3.6, 7., 200)
bb = [analyze(_, z=2., print_stuff=False)[0] for _ in kl]
plt.plot(kl, bb)
plt.xlabel('Kite Length')
plt.ylabel('Blowby')

plt.savefig('plots/blowby-vs-kite-length.png', bbox_inches="tight")
plt.clf()

plt.title("Blowby variation vs Kite length")
kl = np.linspace(0., 4., 20)
bb = [analyze(5.1, z=_, print_stuff=False)[0] for _ in kl]
cm_alpha = [analyze(5.1, z=_, print_stuff=False)[4] for _ in kl]
plt.plot(kl, bb)
plt.plot(kl, cm_alpha)
plt.xlabel('Kite Length')
plt.ylabel('Blowby')

plt.savefig('plots/blowby-vs-z.png', bbox_inches="tight")
plt.clf()