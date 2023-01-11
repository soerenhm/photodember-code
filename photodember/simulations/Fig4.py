# %%
import json
import numpy as np
from scipy.fft import rfft, rfftfreq

from photodember.simulations.mplstyle import *
from photodember.src.constants import SI
from photodember.simulations.core import *


# %%


simulfile = r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_alpha-2.0.dat"
metafile = simulfile.replace(".dat", ".json")

with open(metafile, "r") as io:
    conf = SimulationConfig.from_dict(json.load(io))

init_state = conf.initial_state
simul, _ = conf.create_simulation()
state_t, states = read_simulation_file(simulfile, init_state)


#%%

x = conf.grid
J = np.array(
    [
        electric_field_to_grid(sum(simul.charge_current_density(state)))
        for state in states
    ]
)
E_rad_ampl = (
    1.0 / (4 * np.pi * SI.eps_0 * SI.c_0**2) * np.gradient(J, state_t, axis=0)
)
E_rad_ampl[:, 0] = 0.0  # J = 0
E_rad_ampl[:, -1] = 0.0  # at boundaries

E_rad_tsum = np.trapz(np.abs(E_rad_ampl), x=state_t, axis=0)
F_rad = np.trapz(0.5 * SI.eps_0 * SI.c_0 * E_rad_ampl**2.0, state_t, axis=0)

# %%


def extrapolate_radiation_field(t, rad, t_extr):
    t_extr = t_extr[t_extr > t[-1]]
    opt = fit_exp_decay(0.0, t[-100:], rad[-100:], [rad[-100], 1e-11])
    tail = partial(exp_decay, 0.0, opt.x)
    full = interp1d(np.append(t[:-100], t_extr), np.append(rad[:-100], tail(t_extr)))
    return full


def calc_optical_constants(n, f, th):
    k0 = (2 * np.pi * f) / SI.c_0
    k = n * k0
    eps1 = n**2
    eps2 = 1.0
    kx = np.real(k) * np.sin(th)
    kz1 = np.sqrt(k**2 - kx**2 + 0j)
    kz2 = np.sqrt(k0**2 - kx**2 + 0j)
    kz2 = kz2 if np.imag(kz2) >= 0 else -kz2
    rp = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
    rs = (kz1 - kz2) / (kz1 + kz2)
    tp = 2 * eps2 * kz1 / (eps2 * kz1 + eps1 * kz2) * np.sqrt(eps1 / eps2)
    ts = 2 * kz1 / (kz1 + kz2)
    th1 = th
    if abs(kx) > k0:
        th2 = np.sign(kx) * np.pi / 2
    else:
        th2 = np.arcsin(kx / k0)
    return dict(
        rp=rp, rs=rs, tp=tp, ts=ts, th1=th1, th2=th2, k=k, kx=kx, kz1=kz1, kz2=kz2
    )


def rad_out(tp, ts, E_rad):
    Ep = 0.5 * E_rad * tp
    Es = 0.5 * E_rad * ts
    return Ep, Es


def rad_in(rp, rs, kz, z, E_rad):
    s = np.exp(2j * kz * z)
    Ep = 0.5 * E_rad * (1 + rp * s)
    Es = 0.5 * E_rad * (1 + rs * s)
    return Ep, Es


def total_rad_out(n, f, th, z, rad_ampl):
    c = calc_optical_constants(n, f, th)
    if abs(c["th2"]) >= np.pi / 2:
        return 0j, 0j
    else:
        Ep, Es = rad_out(c["tp"], c["ts"], np.trapz(rad_ampl, z))
        return Ep, Es


def total_rad_in(n, f, th, z, rad_ampl):
    c = calc_optical_constants(n, f, th)

    def calc_field(z, rad_ampl):
        Ep, Es = rad_in(c["rp"], c["rs"], c["kz1"], z, rad_ampl)
        return Ep, Es

    Ep, Es = unzip(map(lambda pair: calc_field(*pair), zip(z, rad_ampl)))
    return np.trapz(Ep, z), np.trapz(Es, z)


# -----------------------------------------------------------------------------
# Calculations for Fig. 4
# %%

# Power-spectral density

t_extr, dt_extr = np.linspace(0.0, 100e-12, 50_000, retstep=True)
E_rad_zsum_extr_fun = extrapolate_radiation_field(
    np.array(state_t), np.trapz(E_rad_ampl, x, axis=1), t_extr
)
E_rad_zsum_extr = E_rad_zsum_extr_fun(t_extr)
E_rad_zsum_fft = rfft(E_rad_zsum_extr)
E_rad_psd = np.abs(E_rad_zsum_fft) ** 2
E_rad_psd /= np.max(E_rad_psd)
f_rad = rfftfreq(len(t_extr), dt_extr)
f_av = np.sum(E_rad_psd * f_rad) / np.sum(E_rad_psd)

# Radiation, direction

n_THz = 2.15  # file:///C:/Users/shm92/Downloads/applsci-11-06733-v2.pdf
f_THz = f_av
th_in = np.linspace(-np.pi / 2, np.pi / 2, 1000)
th_out = np.array([calc_optical_constants(n_THz, f_THz, th)["th2"] for th in th_in])

Epout, Esout = unzip(
    [
        total_rad_out(n_THz, f_THz, th, x, E_rad_tsum * np.abs(np.sin(th)))
        for th in th_in
    ]
)
Epin, Esin = unzip(
    [total_rad_in(n_THz, f_THz, th, x, E_rad_tsum * np.abs(np.sin(th))) for th in th_in]
)
Eout = np.sqrt(np.real(Epout) ** 2 + np.real(Esout) ** 2)
Ein = np.sqrt(np.real(Epin) ** 2 + np.real(Esin) ** 2)


# -----------------------------------------------------------------------------
# Make the figure!
# %%

plt.figure(figsize=(4.5, 5.5))

plt.subplot(2, 1, 1)
plt.plot(t_extr * 1e15, E_rad_zsum_extr * 1e-11, "b-")
plt.ylabel(r"Radiation, $r E / A$ (10$^{11}$ V/m$^2$)")
plt.xlabel("Time, $t$ (fs)")
plt.xlim([0, 500])
plt.ylim([-0.05, 1.6])
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.gca().xaxis.set_minor_locator(MultipleLocator(20))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.4))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))


plt.subplot(2, 2, 3)
plt.plot(f_rad * 1e-12, E_rad_psd, "b-")
plt.xlim([0, 20])
plt.ylim([0, 1.05])
plt.ylabel(f"PSD (arb. units)")
plt.xlabel(f"Frequency, $f$ (THz)")
plt.vlines(f_av * 1e-12, 0, 1.5, linestyle="--", color="k")
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.04))

ax3 = plt.subplot(2, 2, 4, projection="polar")
plt.plot(np.pi - th_in, Ein, "b-")
plt.plot(th_out, Eout, "b-")
# plt.ylabel([])
ax3.set_yticklabels([])
ax3.set_xticklabels([])
ax3.axis("off")

plt.tight_layout()
plt.savefig("Fig4.png", dpi=300, facecolor="w")


# %%

plt.figure()
ax = plt.subplot(1, 1, 1, projection="polar")
plt.plot(np.pi - th_in, Ein, "b-")
plt.plot(th_out, Eout, "b-")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.axis("off")
# plt.axis("off")
# ax.set_thetamin(-90)
# ax.set_thetamax(90)

# total_rad_in(n_THz, f_THz, np.deg2rad(30.0), x, E_rad_tsum)

# -----------------------------------------------------------------------------
# Sanity chekcs... See https://en.wikipedia.org/wiki/Total_internal_reflection
# %%

r = [calc_optical_constants(1.5, 1e15, th) for th in th_in]
rp = np.array([ri["rp"] for ri in r])
rs = np.array([ri["rs"] for ri in r])

idx = [i for i, ri in enumerate(r) if ri["th1"] >= 0.0]
plt.figure()
plt.plot(
    np.rad2deg(th_in)[idx],
    np.clip(np.abs(np.rad2deg(np.angle(rp))[idx]), 0.0, 179.5),
    "r-",
)
plt.plot(np.rad2deg(th_in)[idx], np.abs(np.rad2deg(np.angle(rs))[idx]), "b-")
plt.xlim([0, 90])
plt.ylim([0, 180])
plt.gca().yaxis.set_major_locator(MultipleLocator(20))

# idx = [i for i, ri in enumerate(r) if abs(ri["th2"]) < np.pi / 2]
# plt.plot(np.rad2deg(th_in)[idx], np.array([ri["tp"] for ri in r])[idx])
# plt.plot(np.rad2deg(th_in)[idx], np.array([ri["ts"] for ri in r])[idx])
# plt.plot(np.rad2deg(th_in)[idx], np.array([ri["rp"] for ri in r])[idx])
# plt.plot(np.rad2deg(th_in)[idx], np.array([ri["rs"] for ri in r])[idx])

# %%
