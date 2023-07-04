# %%

from __future__ import annotations
from functools import partial
import json
import numpy as np
from pathlib import Path
import scipy.signal
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

from photodember.src.constants import SI
from photodember.simulations.mplstyle import *
from photodember.simulations.core import (
    electric_field_to_grid,
    exp_decay,
    fit_exp_decay,
    read_simulation_file,
    SimulationConfig,
)


def extrapolate_radiation_field(t, rad, t_extr):
    t_extr = t_extr[t_extr > t[-1]]
    opt = fit_exp_decay(0.0, t[-100:], rad[-100:], [rad[-100], 1e-11])
    tail = partial(exp_decay, 0.0, opt.x)
    full = interp1d(np.append(t[:-100], t_extr), np.append(rad[:-100], tail(t_extr)))
    return full


# %%


def load_current_densities(simulfile: str):
    metafile = simulfile.replace(".dat", ".json")
    with open(metafile, "r") as io:
        conf = SimulationConfig.from_dict(json.load(io))
    init_state = conf.initial_state
    simul, _ = conf.create_simulation()
    state_t, states = read_simulation_file(simulfile, init_state)

    x = conf.grid
    E_D = np.array([electric_field_to_grid(st.electric_field) for st in states])
    J_tot = np.array(
        [
            electric_field_to_grid(sum(simul.charge_current_density(state)))
            for state in states
        ]
    )
    J_tot[:, 0] = 0
    J_tot[:, -1] = 0
    Te = np.array([st.temperature[0] for st in states])
    etae = np.array([st.red_chemical_potential[0] for st in states])
    Th = np.array([st.temperature[1] for st in states])
    etah = np.array([st.red_chemical_potential[1] for st in states])
    L22e = simul.particles[0].L22(Te, etae)
    L22h = simul.particles[1].L22(Th, etah)
    J_D = (L22e + L22h) * E_D
    return J_tot, J_D


def load_current_density_and_peak_density(simulfile: str):
    metafile = simulfile.replace(".dat", ".json")
    with open(metafile, "r") as io:
        conf = SimulationConfig.from_dict(json.load(io))
    init_state = conf.initial_state
    simul, _ = conf.create_simulation()
    state_t, states = read_simulation_file(simulfile, init_state)

    x = conf.grid
    E_D = np.array([electric_field_to_grid(st.electric_field) for st in states])
    J_tot = np.array(
        [
            electric_field_to_grid(sum(simul.charge_current_density(state)))
            for state in states
        ]
    )
    J_tot[:, 0] = 0
    J_tot[:, -1] = 0
    peak_density = max([st.number_density[0][0] for st in states])
    return J_tot, peak_density


thz_files_dir = Path(r"photodember\simulations\thz-scan_new-input-2")
thz_files = list(thz_files_dir.glob("*.dat"))

fluences = []
peak_densities = []
current_densities = []
for thz_file in thz_files:
    filename = thz_file.name
    fluence = float(filename.split("fluence-")[1].split("-Jm-2")[0])
    current_density, peak_density = load_current_density_and_peak_density(str(thz_file))
    fluences.append(fluence)
    current_densities.append(current_density)
    peak_densities.append(peak_density)

# %%


x, dx = np.linspace(0, 1e-6, 1000, retstep=True)
t, dt = np.linspace(0, 2e-12, 2000, retstep=True)

J_tot_THz = [
    scipy.signal.medfilt(np.trapz(J, x, axis=1), 5) for J in current_densities
]  # gets noisy at longer time scales; medfilt helps

# %%

plt.figure(figsize=(5, 4), dpi=300)

# plt.subplot(1, 2, 1)
for i in range(len(J_tot_THz)):
    dJdt = np.gradient(J_tot_THz[i], t)
    density = peak_densities[i]
    # density = 8e27 * (1 / 3) ** thz_files[i][0]
    plt.plot(
        t * 1e15,
        dJdt / density,
        label=f"N = %.1e" % round(density, ndigits=2),
        lw=1,
    )
    # plt.plot(t, dJdt / np.max(np.abs(dJdt)))
plt.legend(frameon=False)
plt.xlim([0, 300])
plt.ylabel(r"$d(J_\mathrm{drift} + J_\mathrm{diff})/dt / N$")
plt.xlabel(r"Time, $t$ (ps)")

plt.tight_layout()
# plt.savefig("THz-rad-vs-density.png", dpi=300)

# %%

plt.figure(figsize=(5, 4), dpi=300)
plt.plot(t * 1e15, np.gradient(J_tot_THz[3], t))
# plt.plot(t * 1e15, np.gradient(J_tot_THz[-1], t))

# %%

dJdt = []
for i in range(len(current_densities)):
    Ne = peak_densities[i]
    n_THz = 2.26  # file:///C:/Users/shm92/Downloads/applsci-11-06733-v2.pdf
    om_THz = 2 * np.pi * 1.8e12
    om_p_av = np.sqrt(SI.e**2 * Ne.mean() / SI.eps_0 / (0.5 * SI.m_e))
    eps_ri = n_THz**2 - om_p_av**2 / (om_THz**2 + 1j * om_THz * 1.6e15)
    n_ri = np.sqrt(eps_ri)
    k = n_ri * om_THz / SI.c_0
    weight = np.exp(-np.imag(k) * x) * np.cos(np.real(k) * x)
    # simple low pass filter... because the derivative of some simulations at intermediate fluences are a bit noisy
    J_int = np.trapz(
        scipy.signal.medfilt2d(current_densities[i] * weight, kernel_size=[5, 1]),
        x,
        axis=1,
    )
    # f = rfftfreq(len(J_int))
    # J_int = rfft(J_int)
    # # J_int[300:] = 0
    # J_int = irfft(J_int)
    # dJdt.append(np.gradient(J, t, axis=0))

    dJdt.append(
        np.gradient(
            J_int,
            t,
            axis=0,
        )
    )
    # dJdt.append(
    #     np.gradient(
    #         scipy.signal.medfilt(np.trapz(current_densities[i] * weight, x, axis=1), 5),
    #         t,
    #         axis=0,
    #     )
    # )

# dJdt = [
#     np.gradient(scipy.signal.medfilt(np.trapz(elem, x, axis=1), 5), t, axis=0)
#     for elem in current_densities
# ]

E_rad = [1.0 / (4 * np.pi * SI.eps_0 * SI.c_0**2) * dJ for dJ in dJdt]

psds = []
f_av = []
f_lo = []
f_hi = []
for E in E_rad:
    t_extr, dt_extr = np.linspace(0.0, 100e-12, 50_000, retstep=True)
    E_rad_extr = extrapolate_radiation_field(np.array(t)[:400], E[:400], t_extr)
    E_fft = rfft(E_rad_extr(t_extr))
    psd = np.abs(E_fft) ** 2
    psd[0] = 0.0  # because of noise, we sometimes get a weird spike at f = 0 THz...
    f_rad = rfftfreq(len(t_extr), dt_extr)
    # f_rad_av = f_rad[np.argmax(psd)]
    psd_dist = psd / np.sum(psd)
    psd_cdf = np.cumsum(psd_dist)

    # find quantiles
    idxs = []
    qs = [0.25, 0.5, 0.75]
    curr = 0
    i = 0
    while len(qs) > 0:
        q = qs.pop(0)
        while i < len(psd_cdf):
            if psd_cdf[i] >= q:
                idxs.append(i)
                break
            i += 1
    f_rad_lo, f_rad_med, f_rad_hi = [f_rad[i] for i in idxs]
    f_rad_av = np.sum(psd_dist * f_rad)
    psds.append((f_rad, psd))
    f_av.append(f_rad_av)
    f_lo.append(f_rad_lo)
    f_hi.append(f_rad_hi)
f_av = np.array(f_av)
f_lo = np.array(f_lo)
f_hi = np.array(f_hi)

thz_energy = [4 * np.pi / 3 * SI.eps_0 * SI.c_0 * np.trapz(E**2, t) for E in E_rad]

# %%

plt.rcParams["xtick.top"] = False

thz_ampls = np.array([np.trapz(elem**2, t) for elem in dJdt])
N = np.array(peak_densities)  # * 8e27
plt.figure(figsize=(5, 4))
# plt.subplot(2, 1, 1)
ax1 = plt.gca()
plt.plot(N, thz_energy, "bo-", fillstyle="none", ms=6)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Excitation density (m$^{-3}$)")
plt.ylabel(r"THz pulse energy (J/m$^4$)", color="b")
# plt.plot(N, 2e-2 * (N / min(N)) ** 3)
plt.yticks(color="b")
# plt.xlim([1e23, 1e28])
# plt.ylim([1e-2, 5e4])

plt.twinx()
# plt.plot(
#     N,
#     [f_rad[np.argmax(psd)] * 1e-12 for (f_rad, psd) in psds],
#     # np.array(f_av) * 1e-12,
#     "sr-",
#     fillstyle="none",
#     zorder=-1,
# )
plt.plot(N, f_av * 1e-12, "sr-", fillstyle="none")
plt.fill_between(N, f_lo * 1e-12, f_hi * 1e-12, color="r", alpha=0.2)
plt.ylim([5.0, 20.0])
plt.ylabel(r"Peak frequency (THz)", color="r", zorder=-1)
plt.yticks(color="r")

f = lambda x: np.interp(x, N, fluences)
g = lambda x: np.interp(x, fluences, N)
ax2 = ax1.secondary_xaxis("top", functions=(f, g))
ax2.set_xlabel(r"Laser fluence (J m$^{-2}$)")
ax2.set_xscale("linear")

plt.tight_layout()
plt.savefig("FigS2.png", dpi=300, facecolor="w", bbox_inches="tight")

# plt.subplot(2,1,2)


# # %%

# simulfile = str(thz_files[-2])
# metafile = simulfile.replace(".dat", ".json")
# with open(metafile, "r") as io:
#     conf = SimulationConfig.from_dict(json.load(io))
# init_state = conf.initial_state
# simul, _ = conf.create_simulation()
# state_t, states = read_simulation_file(simulfile, init_state)

# # %%

# import scipy.signal

# # plt.plot(simul.charge_current_density(states[-1])[1])
# plt.plot(sum(simul.charge_current_density(states[1997])))
# plt.plot(sum(simul.charge_current_density(states[1998])))
# Je = np.array([np.trapz(simul.charge_current_density(st)[0], x=x) for st in states])
# Jh = np.array([np.trapz(simul.charge_current_density(st)[1], x=x) for st in states])

# # %%

# i = 70
# plt.plot(t, -Je, label="electron")
# plt.plot(t, Jh, label="hole")
# plt.plot(t, np.abs(Je + Jh), label="sum")
# # plt.plot(t, -np.array(Jh), label="hole")
# plt.xlim([0e-15, 200e-15])
# plt.yscale("log")
# # plt.ylim([1e-4, 1e5])
# plt.legend(frameon=False)
# # plt.plot(x, simul.charge_current_density(states[i + 1])[0])

# # %%

# plt.plot(t, scipy.signal.medfilt(np.gradient(J1, axis=0), 5))
# # %%

# %%
