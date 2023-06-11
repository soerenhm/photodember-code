# %%
import json
import numpy as np

from photodember.simulations.mplstyle import *
from photodember.src.constants import SI
from photodember.simulations.core import *


# Load simulation
# -----------------------------------------------------------------------------
# %%

simulfile = r"photodember\simulations\SiO2_fluence_tau-120fs.dat"
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
plt.plot(x_nm, electron_states[peak_density_tzero].number_density * 1e-27, "k-")
plt.plot(x_nm, Esim[peak_density_tzero, :] * 3e-7, "g--")
plt.xlim([0, 1000])
plt.ylim([0, 6.0])
plt.xlabel(r"Coordinate, $z$ (nm)")
plt.ylabel(r"Density, $N$ (10$^{27}$ m$^{-3}$)")
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.2))
plt.gca().xaxis.set_major_locator(MultipleLocator(200))
plt.gca().xaxis.set_minor_locator(MultipleLocator(40))
plt.annotate(r"$N$", xy=(300, 2.0), fontsize="large", color="k")
plt.annotate(r"$T_e$", xy=(60, 5.0), fontsize="large", color="b")
plt.annotate(r"$T_h$", xy=(120, 4.2), fontsize="large", color="r")
plt.annotate(r"$E_D$", xy=(180, 1.6), fontsize="large", color="g")

ax = plt.twinx()
plt.plot(x_nm, electron_states[peak_density_tzero].temperature * 1e-4, "b-")
plt.plot(x_nm, hole_states[peak_density_tzero].temperature * 1e-4, "r-")
plt.ylabel(r"Temperature, $T_i$ (10$^{4}$ K)")
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.ylim([1.4, 2.2])

plt.subplot(2, 1, 2)
t_fs = np.array(t) * 1e15
plt.plot(t_fs_offset, Esim[:, 0] * 1e-7, "g-")

plt.fill_between(
    t_fs_offset,
    0.12
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
plt.ylim([0, 0.6])
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))
plt.annotate(r"$E_D$", xy=(210, 0.10), fontsize="large", color="g")
plt.annotate(r"$T_e$", xy=(60, 0.53), fontsize="large", color="b")
plt.annotate(r"$T_h$", xy=(100, 0.44), fontsize="large", color="r")

plt.twinx()
Te_surf = [st.temperature[0] * 1e-4 for st in electron_states]
Th_surf = [st.temperature[0] * 1e-4 for st in hole_states]
plt.plot(t_fs_offset, Te_surf, "b-")
plt.plot(t_fs_offset, Th_surf, "r-")
plt.ylabel(r"Temperature, $T_i$ (10$^4$ K)")
plt.xlim([-100, 1000])
plt.ylim([0, 2.4000])
plt.gca().xaxis.set_minor_locator(MultipleLocator(25))
plt.gca().yaxis.set_minor_locator(MultipleLocator(400 * 1e-4))
plt.gca().yaxis.set_major_locator(MultipleLocator(4000 * 1e-4))

plt.tight_layout()
fig.savefig("Fig2a.png", dpi=300, facecolor="w", bbox_inches="tight")

# %%
