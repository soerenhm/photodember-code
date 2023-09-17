# %%

from __future__ import annotations
from collections import namedtuple
from functools import partial
import json
import numpy as np
from pathlib import Path
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


def load_current_density_and_peak_density(simulfile: str):
    metafile = simulfile.replace(".dat", ".json")
    with open(metafile, "r") as io:
        conf = SimulationConfig.from_dict(json.load(io))
    init_state = conf.initial_state
    simul, _ = conf.create_simulation()
    _, states = read_simulation_file(simulfile, init_state)

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


def calculate_dJdt(x, current_density, peak_density):
    Ne = peak_density
    n_THz = 2.26  # file:///C:/Users/shm92/Downloads/applsci-11-06733-v2.pdf
    om_THz = 2 * np.pi * 1.8e12
    om_p_av = np.sqrt(SI.e**2 * Ne.mean() / SI.eps_0 / (0.5 * SI.m_e))
    eps_ri = n_THz**2 - om_p_av**2 / (om_THz**2 + 1j * om_THz * 1.6e15)
    n_ri = np.sqrt(eps_ri)
    k = n_ri * om_THz / SI.c_0
    weight = np.exp(-np.imag(k) * x) * np.cos(np.real(k) * x)

    # running mean is much better!
    window = 11
    J_int = np.trapz(current_density * weight, x, axis=1)
    J_int2 = np.append(
        J_int[: window - 1],
        np.convolve(J_int, np.ones(window) / window, mode="valid"),
    )
    return np.gradient(J_int2, t, axis=0)


def calculate_Erad(dJdt):
    return 1.0 / (4 * np.pi * SI.eps_0 * SI.c_0**2) * dJdt


def calculate_rad_energy(t, E):
    return 4 * np.pi / 3 * SI.eps_0 * SI.c_0 * np.trapz(E**2, t)


def calculate_psd(t, E):
    t_extr, dt_extr = np.linspace(0.0, 100e-12, 50_000, retstep=True)
    E_rad_extr = extrapolate_radiation_field(np.array(t)[:800], E[:800], t_extr)
    E_fft = rfft(E_rad_extr(t_extr))
    psd = np.abs(E_fft) ** 2
    psd[0] = 0.0  # because of noise, we sometimes get a weird spike at f = 0 THz...
    f_rad = rfftfreq(len(t_extr), dt_extr)
    # f_rad_av = f_rad[np.argmax(psd)]
    psd_dist = psd / np.sum(psd)
    return psd_dist, f_rad


def find_quantile_indices(cdf, qs):
    idxs = []
    i = 0
    qs = sorted(qs)
    while len(qs) > 0:
        q = qs.pop(0)
        while i < len(cdf):
            if cdf[i] >= q:
                idxs.append(i)
                break
            i += 1
    return idxs


def cov(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.mean(x * y) - np.mean(x) * np.mean(y)


def straight_line_slope(x, y):
    return cov(x, y) / cov(x, x)


# %%

FluenceScan = namedtuple(
    "PulseDurScan", ["pulse_dur", "fluence", "peak_density", "current_density"]
)

thz_files_dir = Path(r"photodember\simulations\thz-scan_new-input-2")
thz_files = list(thz_files_dir.glob("*.dat"))

fluence_scan = FluenceScan(35e-15, [], [], [])
for thz_file in thz_files:
    filename = thz_file.name
    fluence = float(filename.split("fluence-")[1].split("-Jm-2")[0])
    current_density, peak_density = load_current_density_and_peak_density(str(thz_file))
    fluence_scan.fluence.append(fluence)
    fluence_scan.current_density.append(current_density)
    fluence_scan.peak_density.append(peak_density)


# %%

PulseDurScan = namedtuple(
    "PulseDurScan", ["fluence", "pulse_dur", "peak_density", "current_density"]
)

thz_files_dir = Path(r"photodember\simulations\thz-scan_pulse-dur")
thz_files = list(thz_files_dir.glob("*.dat"))

pulse_dur_scan = PulseDurScan(1.6, [], [], [])
for thz_file in thz_files:
    filename = thz_file.name
    pulse_dur = float(filename.split("dur_")[1].split("-fs")[0])
    current_density, peak_density = load_current_density_and_peak_density(str(thz_file))
    pulse_dur_scan.pulse_dur.append(pulse_dur)
    pulse_dur_scan.current_density.append(current_density)
    pulse_dur_scan.peak_density.append(peak_density)


# %%

x, dx = np.linspace(0, 1e-6, 1000, retstep=True)
t, dt = np.linspace(0, 2e-12, 2000, retstep=True)


def get_psd(x, t, current_density, peak_density):
    dJdt = calculate_dJdt(x, current_density, peak_density)
    return calculate_psd(t, calculate_Erad(dJdt))


def get_psd_stats(psd, freqs):
    q = [0.25, 0.50, 0.75]
    inds = [np.argmax(psd)] + find_quantile_indices(np.cumsum(psd), q)
    fq = [freqs[i] for i in inds]
    return fq


# %%

plt.rcParams["xtick.top"] = False
import matplotlib.ticker as ticker

fig, axs = plt.subplots(figsize=(8, 4), ncols=2, dpi=300)
ax1, ax2 = axs

# =====================================
# THZ energy subplot

thz_energy = np.array(
    [
        calculate_rad_energy(t, calculate_Erad(calculate_dJdt(x, J, N)))
        for J, N in zip(fluence_scan.current_density, fluence_scan.peak_density)
    ]
)
N = np.array(fluence_scan.peak_density)  # * 8e27
F = np.array(fluence_scan.fluence)
psd_stats = (
    np.array(
        [
            get_psd_stats(*get_psd(x, t, J, N))
            for (J, N) in zip(fluence_scan.current_density, fluence_scan.peak_density)
        ]
    )
    * 1e-12
)
f_pk = psd_stats[:, 0]
f_lo = psd_stats[:, 1]
f_hi = psd_stats[:, 3]

plt.sca(ax1)
plt.plot(N, thz_energy, "bo-", fillstyle="none", ms=6)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Excitation density, $N$ (m$^{-3}$)")
plt.ylabel(r"THz pulse energy, $\mathcal{E}$ (J/m$^4$)", color="b")
plt.yticks(color="b")
plt.ylim([1e2, 5e4])

plt.twinx()
plt.plot(N, psd_stats[:, 2], "sr-", fillstyle="none")
plt.fill_between(N, f_lo, f_hi, color="r", alpha=0.2)
plt.ylim([5.0, 20.0])
plt.ylabel(r"Peak frequency, $f$ (THz)", color="r", zorder=-1)
plt.yticks(color="r")
f = lambda x: np.interp(x, N, F)
g = lambda x: np.interp(x, F, N)
ax12 = ax1.secondary_xaxis("top", functions=(f, g))
ax12.set_xlabel(r"Laser fluence, $F$ (J cm$^{-2}$)")
ax12.set_xscale("linear")

# =====================================
# THZ pulsewidth subplot

plt.sca(ax2)

t_pulse = np.array(pulse_dur_scan.pulse_dur) * 1e15
bw_pulse = 0.44 / t_pulse * 1e3
I = sorted(
    range(len(bw_pulse)),
    key=lambda i: bw_pulse[i],
)
bw_pulse = bw_pulse[I]
t_pulse = t_pulse[I]
psd_stats = (
    np.array(
        [
            get_psd_stats(*get_psd(x, t, J, N))
            for (J, N) in zip(
                pulse_dur_scan.current_density, pulse_dur_scan.peak_density
            )
        ]
    )[I]
    * 1e-12
)

err_lo = psd_stats[:, 2] - psd_stats[:, 1]
err_hi = psd_stats[:, 3] - psd_stats[:, 2]
slope = straight_line_slope(bw_pulse[:3], psd_stats[:3, 0])
plt.plot(bw_pulse, psd_stats[:, 0], "ro-", fillstyle="none")
plt.fill_between(
    bw_pulse, psd_stats[:, 0] - err_lo, psd_stats[:, 0] + err_hi, color="r", alpha=0.2
)
x_ = np.linspace(0, 50, 10)
y_ = slope * x_
ax2.plot(x_, y_, "k--", zorder=-1)
plt.ylim([0, 50])
plt.xlim([2, 45])
plt.ylabel(r"Peak frequency, $f$ (THz)")
plt.xlabel(r"Excitation bandwidth, $0.44/\tau$ (THz)")

f = lambda x: 0.44 / (x + 1e-12) * 1e3
g = f
ax22: plt.Axes = ax2.secondary_xaxis(
    "top",
    functions=(f, g),
)
ax22.set_xlim([f(1), f(50)])
ax22.set_xlabel(r"Pulse duration, $\tau$ (fs)")
ax22.xaxis.set_major_locator(ticker.FixedLocator([100, 50, 30, 20, 10]))

fig.tight_layout()
fig.savefig("FigS2a", dpi=300)

# %%

dember_mag = np.array(
    [
        calculate_Erad(calculate_dJdt(x, J, N))
        for J, N in zip(pulse_dur_scan.current_density, pulse_dur_scan.peak_density)
    ]
)
