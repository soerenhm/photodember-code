# %%
import json
import numpy as np

from photodember.simulations.mplstyle import *
from photodember.src.constants import SI
from photodember.simulations.core import *


# Load simulation
# -----------------------------------------------------------------------------
# %%

simulfile = r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_alpha-2.0.dat"
metafile = simulfile.replace(".dat", ".json")

with open(metafile, "r") as io:
    simulconf = SimulationConfig.from_dict(json.load(io))

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

peak_density_tzero = np.argmax([st.number_density[0] for st in electron_states])


# FIGURE 1
# -----------------------------------------------------------------------------
# %%

x_nm = x * 1e9
ts = [0.0, 10e-15, 100e-15, 1e-12]
Esim = np.copy(electric_field)
ids = [np.argmin(np.abs(np.array(t) - ti)) for ti in ts]
lines = ["-", "--", "-.", ":"]

fig = plt.figure(figsize=(4.5, 5.5), dpi=125)
plt.subplot(2, 1, 1)
plt.plot(x_nm, electron_states[peak_density_tzero].number_density * 1e-28, "k-")
plt.xlim([0, 1000])
plt.ylim([0, 1.0])
plt.xlabel(r"Coordinate, $x$ (nm)")
plt.ylabel(r"Density, $N$ (10$^{28}$ m$^{-3}$)")
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.04))
plt.gca().xaxis.set_major_locator(MultipleLocator(200))
plt.gca().xaxis.set_minor_locator(MultipleLocator(40))
plt.annotate(r"$N$", xy=(110, 0.3), fontsize="large", color="k")
plt.annotate(r"$T_e$", xy=(50, 0.65), fontsize="large", color="b")
plt.annotate(r"$T_h$", xy=(160, 0.48), fontsize="large", color="r")

ax = plt.twinx()
plt.plot(x_nm, electron_states[peak_density_tzero].temperature, "b-")
plt.plot(x_nm, hole_states[peak_density_tzero].temperature, "r-")
plt.ylabel(r"Temperature, $T_i$ (K)")
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.yaxis.set_minor_locator(MultipleLocator(40))
plt.ylim([3400, 4400])

plt.subplot(2, 1, 2)
t_fs = np.array(t) * 1e15
plt.plot(t_fs_offset, Esim[:, 0] * 1e-7, "g-")

plt.fill_between(
    t_fs_offset,
    0.1
    * np.exp(
        -4
        * np.log(2)
        * (
            (np.array(t) - simulconf.excitation_time_zero)
            / simulconf.excitation_time_fwhm
        )
        ** 2
    ),
    lw=1,
    ls="-",
    color="k",
    hatch="///",
    fc="k",
    alpha=0.33,
    zorder=-3,
)
# plt.xscale("log")
plt.xlabel(r"Time, $t$ (fs)")
plt.ylabel(r"Electric field, $E_D$ (10$^7$ V m$^{-1}$)")
plt.ylim([0, 0.5])
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))
plt.annotate(r"$E_D$", xy=(210, 0.21), fontsize="large", color="g")
plt.annotate(r"$T_e$", xy=(60, 0.43), fontsize="large", color="b")
plt.annotate(r"$T_h$", xy=(100, 0.34), fontsize="large", color="r")

plt.twinx()
Te_surf = [st.temperature[0] for st in electron_states]
Th_surf = [st.temperature[0] for st in hole_states]
plt.plot(t_fs_offset, Te_surf, "b-")
plt.plot(t_fs_offset, Th_surf, "r-")
plt.ylabel(r"Temperature, $T_i$ (K)")
plt.xlim([-100, 1000])
plt.ylim([0, 5100])
plt.gca().xaxis.set_minor_locator(MultipleLocator(25))
plt.gca().yaxis.set_minor_locator(MultipleLocator(200))
plt.gca().yaxis.set_major_locator(MultipleLocator(1000))

plt.tight_layout()
# fig.savefig("Fig2.png", dpi=300, facecolor="w")
# %%


plt.pcolormesh(t * 1e15, x * 1e9, electric_field.T, cmap=cc.cm["fire"])
plt.colorbar()
plt.ylim([0, 50])
plt.xlim([0, 1000])
