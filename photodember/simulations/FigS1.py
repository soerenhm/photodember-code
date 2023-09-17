# %%
import json
import numpy as np

from photodember.simulations.mplstyle import *
from photodember.src.constants import SI
from photodember.simulations.core import *


# Load simulation
# -----------------------------------------------------------------------------
# %%

simulfile = r"photodember/simulations/SiO2_fluence_tau-120fs.dat"
metafile = simulfile.replace(".dat", ".json")

with open(metafile, "r") as io:
    meta = json.load(io)
    simulconf = SimulationConfig.from_dict(meta)

init_state = simulconf.initial_state
simul, _ = simulconf.create_simulation()
t, states = read_simulation_file(simulfile, init_state)


# Prepare variables
# -----------------------------------------------------------------------------
# %%


x = np.copy(simulconf.grid)
t = np.array(t)
t_fs = t * 1e15
t_fs_offset = t_fs - simulconf.excitation_time_zero * 1e15
electron_states = [st.particle(0) for st in states]
hole_states = [st.particle(1) for st in states]
electric_field = np.array([electric_field_to_grid(st.electric_field) for st in states])

Ne = np.array([st.number_density[0] for st in states])
Nh = np.array([st.number_density[1] for st in states])
Te = np.array([st.temperature[0] for st in states])

peak_density_tzero = np.argmax([st.number_density[0] for st in electron_states])


# %%

x_nm = x * 1e9
t_fs = (t - simulconf.excitation_time_zero) * 1e15
t_plot = [50, 200, 2000]
ti = [np.argmin(np.abs(t_fs - ti)) for ti in t_plot]

fig, axs = plt.subplots(figsize=(12, 4), ncols=2, sharex=True)

ax2 = axs[0]
ax22 = ax2.twinx()
ax3 = axs[1]

ls = ["-", "--", ":"]
for i, ti_ in enumerate(ti):
    ax2.plot(
        x_nm, Ne[ti_, :] * 1e-27, "k", linestyle=ls[i], label=f"t = {t_plot[i]} fs"
    )
    ax22.plot(x_nm, SI.e * (Nh[ti_, :] - Ne[ti_, :]) * 1e-5, "b", linestyle=ls[i])
    ax3.plot(x_nm, Te[ti_, :], "r", linestyle=ls[i])

for ax in (ax2, ax3):
    ax.set_xlim([1, 1000])
    ax.set_xscale("log")
    ax.set_xlabel(r"Depth, $z$ (nm)")
ax2.set_ylim([0.0, 6])
ax3.set_ylim([5000.0, 20500])
ax22.set_ylim([-0.2, 1.8])
ax2.legend(frameon=False)
ax2.set_ylabel(r"Electron density, $N_e$ (10$^{27}$ m$^{-3}$)")
ax22.set_ylabel(r"Charge density, $\rho$ (10$^{5}$ C m$^{-3}$)", color="b")
plt.sca(ax22)
plt.xticks(color="b")
ax3.set_ylabel(r"Electron temperature, $T_e$ (K)")

fig.tight_layout()
fig.savefig("FigS1a.png", dpi=300, facecolor="w", bbox_inches="tight")

# %%
