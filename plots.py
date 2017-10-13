import matplotlib.pyplot as plt
import numpy as np
import seaborn
from common import Tether
from analyzer import analyze
#
# alpha = np.linspace(0., 0.5, 5)
# plt.title('Cl vs alpha (rad)')
# plt.plot(alpha, Clk(alpha), marker='o')
# plt.savefig('plots/kite-cl.png', bbox_inches="tight")
# plt.clf()
#
# plt.title('Cd vs alpha (rad)')
# plt.plot(alpha, Cdk(alpha), marker='o')
# plt.savefig('plots/kite-cd.png', bbox_inches="tight")
# plt.clf()
#
# plt.title('Cm vs alpha (rad)')
# plt.plot(alpha, Cmk(alpha), marker='o')
# plt.savefig('plots/kite-cm.png', bbox_inches="tight")
# plt.clf()
#
# tether = Tether(100., 1.)
# plt.title('Tether profile')
# tether.plot_profile(100., 1081)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('plots/tether-profile.png', bbox_inches="tight")
# plt.clf()
#
# plt.title("Blowby variation vs Kite length")
# kl = np.linspace(3.6, 7., 200)
# bb = [analyze(_, z=2., print_stuff=False)[0] for _ in kl]
# plt.plot(kl, bb)
# plt.xlabel('Kite Length')
# plt.ylabel('Blowby')
#
# plt.savefig('plots/blowby-vs-kite-length.png', bbox_inches="tight")
# plt.clf()
#
# # plt.title("Blowby variation vs Kite length")
# # kl = np.linspace(0., 4., 20)
# # bb = [analyze(5.1, z=_, print_stuff=False)[0] for _ in kl]
# # cm_alpha = [analyze(5.1, z=_, print_stuff=False)[4] for _ in kl]
# # plt.plot(kl, bb)
# # plt.plot(kl, cm_alpha)
# # plt.xlabel('Kite Length')
# # plt.ylabel('Blowby')
# #
# # plt.savefig('plots/blowby-vs-z.png', bbox_inches="tight")
# # plt.clf()
#
# lk = np.linspace(0., 10., 100)
# plt.title("Equilibrium angle of attack vs kite root chord length")
# aoa = np.zeros_like(lk)
# cm_alpha = np.zeros_like(lk)
# for i in range(len(lk)):
#     aerod_data = analyze(lk[i], 1.2, 1.5)[4]
#     aoa[i] = aerod_data['alpha']
#     cm_alpha[i] = aerod_data['Cm_alpha']
#
# plt.plot(lk, aoa, label=r"$\alpha$")
# plt.plot(lk, cm_alpha, label=r"$c_{m_\alpha}$")
# plt.legend(loc='best')
# plt.savefig('plots/aoa-vs-lk.png', bbox_inches='tight')
# plt.clf()
#

free_lift = np.linspace(15, 500, 100)
blowby = []
tether_stress = []
for i in range(len(free_lift)):
    aerostat, es, ts = analyze(free_lift=free_lift[i])
    blowby.append(aerostat.blowby)
    tether_stress.append(ts / 1e6)
plt.plot(free_lift, blowby)
plt.title('Blowby vs free lift')
# plt.plot(free_lift, tether_stress, label='Tether Stress')
# plt.legend(loc='best')
plt.xlabel('Free Lift (%)')
plt.ylabel('Blowby (m)')
plt.savefig('plots/blowby-vs-free-lift.png', bbox_inches='tight')
plt.clf()

plt.title('Tether stress vs free lift')
plt.plot(free_lift, tether_stress)
plt.xlabel('Free Lift (%)')
plt.ylabel('Tether Stress (MPa)')
plt.savefig('plots/tether-stress-vs-free-lift.png', bbox_inches='tight')
plt.clf()