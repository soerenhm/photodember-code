# %%
import json
import numpy as np
from scipy.interpolate import CubicSpline

from photodember.simulations.simulator import DielectricConfig, read_simulation_file

# Pimp my plot
# -----------------------------------------------------------------------------
# %%

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['animation.ffmpeg_path'] = "C:/ffmpeg/bin/ffmpeg.exe"

# Load simulation
# -----------------------------------------------------------------------------
# %%

simulfile = "photodember/simulations/fs1_source_term_with_T_rise.dat"
metafile = simulfile.replace(".dat", ".json")

with open(metafile, "r") as io:
    simulconf = DielectricConfig.from_dict(json.load(io))

init_state = simulconf.initial_state
simul, _ = simulconf.create_simulation()
t, states = read_simulation_file(simulfile, init_state)

# Prepare variables
# -----------------------------------------------------------------------------
# %%

x = simulconf.grid
t_fs = np.array(t) * 1e15
electron_states = [st.particle(0) for st in states]
hole_states = [st.particle(1) for st in states]
electric_field = np.array([st.electric_field for st in states])

peak_density_tzero = np.argmax([st.number_density[0] for st in electron_states])

# %% Centroid

t_s = np.array(t)
electron_density = np.array([st.number_density[0] for st in states])

exprep0 = lambda x, N: np.repeat(np.expand_dims(x, 0), N, 0)
exprep1 = lambda x, N: np.repeat(np.expand_dims(x, 1), N, 1)

X = exprep0(x, len(t))
Pr = electron_density / exprep1(np.sum(electron_density, axis=1), len(x))
mean_X = np.sum(X * Pr, axis=1)
var_X = np.sum((X - exprep1(mean_X, len(x)))**2 * Pr, axis=1)

ids = np.array(t) > conf.excitation_time_zero

plt.figure(figsize=(5,4.5))
# plt.plot(t_fs[ids], (mean_X[ids]-mean_X[ids][0]) * 1e9)
plt.plot(t_fs[ids], np.sqrt(var_X[ids]) * 1e9)
plt.xlabel(r"Time, $t$ (fs)")
plt.ylabel(r"Centroid, $\Delta x$ (nm)")
# plt.ylim([0, 2.1])
plt.xlim([0, 5000])
# plt.xscale("log")
# plt.yscale("log")
plt.tight_layout()
# np.mean(electron_density)

# %%

surface_electric_field = [st.electric_field[1]*.5 for st in states]

plt.plot(t_fs, surface_electric_field, "g-")
plt.xlim([0, 1000])
plt.ylim([0, 1.6e7])

# FIGURE 1
# -----------------------------------------------------------------------------
# %%

x_nm = x * 1e9
ts = [0.0, 10e-15, 100e-15, 1e-12]
Esim = np.copy(electric_field)
ids = [np.argmin(np.abs(np.array(t)-ti)) for ti in ts]
lines = ["-", "--", "-.", ":"]

fig = plt.figure(figsize=(4.5,5.5), dpi=125)
plt.subplot(2,1,1)
plt.plot(x_nm, electron_states[peak_density_tzero].number_density * 1e-28, "k-")
plt.xlim([0, 1000])
plt.ylim([0, 1])
plt.xlabel(r"Coordinate, $x$ (nm)")
plt.ylabel(r"Density, $N$ (10$^{28}$ m$^{-3}$)")
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.04))
plt.annotate(r"$N$", xy=(30, 0.7), fontsize="large", color="k")
plt.annotate(r"$T_e$", xy=(200, 0.88), fontsize="large", color="b")
plt.annotate(r"$T_h$", xy=(180, 0.65), fontsize="large", color="r")
ax = plt.twinx()
plt.plot(x_nm, electron_states[peak_density_tzero].temperature, "b-")
plt.plot(x_nm, hole_states[peak_density_tzero].temperature, "r-")
# plt.plot(x_nm, electron_states[peak_density_tzero].temperature, "b-")
# plt.plot(x_nm, hole_states[peak_density_tzero].temperature, "r--")
plt.ylim([2300, 4000])
plt.ylabel(r"Temperature, $T_i$ (K)")
ax.yaxis.set_major_locator(MultipleLocator(300))
ax.yaxis.set_minor_locator(MultipleLocator(100))

plt.subplot(2,1,2)
t_fs = np.array(t) * 1e15
plt.plot(t_fs, .5*(Esim[:,0]+Esim[:,1])*1e-7, "g-")
plt.fill_between(t_fs, 1.3*np.exp(-4*np.log(2)*((np.array(t)-conf.excitation_time_zero)/conf.excitation_time_fwhm)**2), lw=1, ls="-", color="k", hatch='///', fc="k", alpha=0.33, zorder=-3)
# plt.xscale("log")
plt.xlabel(r"Time, $t$ (fs)")
plt.ylabel(r"Electric field, $E_D$ (10$^7$ V m$^{-1}$)")
plt.ylim([0, 1.6])
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
plt.annotate(r"$E_D$", xy=(270, 0.7), fontsize="large", color="g")
plt.annotate(r"$T_e$", xy=(200, 1.35), fontsize="large", color="b")
plt.annotate(r"$T_h$", xy=(230, 1.1), fontsize="large", color="r")

plt.twinx()
Te_surf = [st.temperature[0] for st in electron_states]
Th_surf = [st.temperature[0] for st in hole_states]
plt.plot(t_fs, Te_surf, "b-")
plt.plot(t_fs, Th_surf, "r-")
plt.ylabel(r"Temperature, $T_i$ (K)")
plt.xlim([0, 1e3])
plt.ylim([0, 3500])
plt.gca().xaxis.set_minor_locator(MultipleLocator(25))
plt.gca().yaxis.set_minor_locator(MultipleLocator(100))
plt.gca().yaxis.set_major_locator(MultipleLocator(500))

plt.tight_layout()
# fig.savefig("Fig2.png", dpi=300, facecolor="w")


# Semi-classical equation of motion
# -----------------------------------------------------------------------------
# %% Semi-classical equation of motion

dt = t[1]-t[0]
# E = np.array([simul.ambipolar_electric_field(st) for st in states])
k = -SI.e/SI.hbar * np.cumsum(electric_field*dt, axis=0)

a = 4.9e-10 # https://link.springer.com/content/pdf/10.1007/BF00552441.pdf
kBz = np.pi/a

x0 = 0.0 # nm
ix = np.argmin(np.abs(x - x0))

plt.figure(figsize=(5,4), dpi=150)
plt.plot(x_nm, -k[400,:]/kBz, "g-", label="t = 0.2 ps")
plt.plot(x_nm, -k[1000,:]/kBz, "c-", label="t = 0.6 ps")
plt.plot(x_nm, -k[-1,:]/kBz, "m-", label="t = 1.0 ps")
plt.legend(frameon=False)
plt.ylabel(r"Momentum, $\Delta k/k_{\mathrm{BZ}}$")
plt.xlabel("Coordinate, x (nm)")
plt.xlim([0, 500])
# plt.ylim([0, 0.07])
plt.tight_layout()


# Band structure
# -----------------------------------------------------------------------------
# %%

def energy_vs_k(alpha, m, kx, ky, kz):
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    q = SI.hbar*k/np.sqrt(m)
    return (-1 + np.sqrt(1 + 2*alpha*q**2))/(2*alpha)

def effective_mass(m, alpha, kx):
    q = SI.hbar * kx / np.sqrt(m)
    return m * (1 + 2*alpha*q**2)**(3/2)

ky = 0.0; kz = 0.0
kx = np.linspace(-0.8, 0.8, 200) * kBz
alpha = 1.0/SI.e
mc = 0.5*SI.m_e
ek = energy_vs_k(alpha, mc, kx, ky, kz)

tau_eph = 100e-15
dt = t[1]-t[0]
dt_array = np.gradient(t)
conv = np.exp(-np.array(t)/tau_eph) * dt_array
electric_field_t_integral = np.array([np.convolve(conv, electric_field[:,i])[:len(t)] for i in range(0, electric_field.shape[1])])


# Helmholtz....
# -----------------------------------------------------------------------------
# %%

from photodember.src.optics.helmholtz import solve_stack, Stack, Layer
from photodember.src.optics.optical_models import Drude
from scipy.interpolate import interp1d

eps_v = 3.11
n_val = 3.2e29
mc = 0.5 * SI.m_e
probe_om = 800/700 * 2.35e15
alpha = 0.2 / SI.e
xgrid = np.copy(x)
dx = xgrid[1]-xgrid[0]

def create_eps_from(om, density, mass, gam):
    def eps(x):
        nx = density(x)
        om_p = np.sqrt(SI.e**2*nx/(SI.eps_0*mass(x)))
        return eps_v - (eps_v - 1)*(nx/n_val) - om_p**2/(om*(om + 1j*gam(x)))
    return eps

state_index = 100

def solve_helmholtz(alpha, gam, state_index):
    state = electron_states[state_index]
    k_z = interp1d(xgrid, SI.e * electric_field_t_integral[:, state_index] / SI.hbar, kind="cubic")
    mz = lambda x: effective_mass(mc, alpha, k_z(x))
    gx = lambda x: gam

    eps_x = create_eps_from(probe_om, interp1d(xgrid, state.number_density), lambda x: mc, gx)
    eps_z = create_eps_from(probe_om, interp1d(xgrid, state.number_density), mz, gx)
    stack_s = Stack([
        Layer.create(xgrid[-1]-dx, eps_x), 
        Layer.create(np.inf, lambda _: eps_v + 0j)])
    stack_p = Stack([
        Layer.create(xgrid[-1]-dx, eps_z), 
        Layer.create(np.inf, lambda _: eps_v + 0j)])

    ss = solve_stack(stack_s, probe_om, 60.0, "s")
    sp = solve_stack(stack_p, probe_om, 60.0, "p")
    return ss, sp

state_inds = list(range(0, 200, 50)) + list(range(210, 390, 10)) + list(range(400, 1000, 50)) + [len(t)-1]
alphas_int = [0.1, 0.2, 0.4]
alphas = np.array(alphas_int) / SI.e
gam = 1e15
hhsols = [[solve_helmholtz(alpha, gam, i) for i in state_inds] for alpha in alphas]

# THz....
# -----------------------------------------------------------------------------
# %%

skindepth_THz = 1e-9
dx = x[1]-x[0]
THz_averager = np.repeat(np.expand_dims(np.exp(-(x+0.5*dx)/skindepth_THz) * dx, 0), len(t), axis=0)
J_tot = np.array([sum(simul.charge_current_density(state)) for state in states])
J_THz_ = CubicSpline(t, np.sum(J_tot * THz_averager, axis=1))#, k=3, s=len(t))
J_THz = J_THz_(t)
# dJdt = np.gradient(J_THz, t, axis=0)
# d2Jdt2 = np.gradient(dJdt, t, axis=0)
dJdt = J_THz_(t, 1)
d2Jdt2 = J_THz_(t, 2)
Ens = SI.mu_0/(6*np.pi*SI.c_0) * d2Jdt2
Prad = -Ens * J_THz

plt.plot(t_fs, Prad)
# plt.xlim([0, 1500])


# FIGURE 3
# -----------------------------------------------------------------------------
# %%

delta_k = SI.e * electric_field_t_integral[1,:] / SI.hbar / kBz
Ne = np.array([st.number_density for st in electron_states]).T
skindepth = 40 # nm; good until about 40 nm...
p = np.exp(-x_nm/skindepth); p /= np.sum(p)
P = np.repeat(np.expand_dims(p, 1), len(t), axis=1)
t_offset = np.array(t) - simulconf.excitation_time_zero #- t[peak_density_tzero]


fig = plt.figure(figsize=(4.5,9.0), dpi=125)
plt.subplot(4,1,1)
kxt = np.sum(delta_k * P, axis=0)
plt.plot(np.array(t_offset)*1e15, kxt, "k-")
plt.xlim([-150, 850])
plt.ylim([0, 0.4])
# plt.xscale("log")
plt.ylabel(r"Momentum, $\Delta k / k_{\mathrm{BZ}}$")
plt.xlabel(r"Time, $t$ (fs)")

plt.subplot(4,1,2)
ls = ["c-", "m--", "y-."]
for i, alpha in enumerate(alphas_int):
    mxt = np.interp(kxt*kBz, kx, effective_mass(mc, alpha/SI.e, kx)) / mc
    # mx = effective_mass(alpha/SI.e, kx) / mc
    plt.plot(np.array(t_offset)*1e15, mxt, ls[i])
plt.xlabel("")
plt.xlim([0, 800])
plt.ylim([1, 2.25])
plt.annotate(r"$\alpha$ = 0.1 eV$^{-1}$", xy=(140, 1.3), color="c")
plt.annotate(r"$\alpha$ = 0.2 eV$^{-1}$", xy=(170, 1.6), color="m")
plt.annotate(r"$\alpha$ = 0.4 eV$^{-1}$", xy=(270, 1.9), color="y")
plt.xlabel(r"Time, $t$ (fs)")
plt.ylabel(r"Effective mass, $m / m_c$")

Rps = []
for hhsola in hhsols:
    Rp_ = [elem[1].R for elem in hhsola]
    Rps.append(Rp_)

plt.subplot(4,1,3)
# Reflection, transmission
plt.plot(t_offset[state_inds]*1e15, Rps[0], "c-")
plt.plot(t_offset[state_inds]*1e15, Rps[1], "m--")
plt.plot(t_offset[state_inds]*1e15, Rps[2], "y-.")
plt.annotate(r"$\alpha$ = 0.1 eV$^{-1}$", xy=(250, 0.13), color="c")
plt.annotate(r"$\alpha$ = 0.2 eV$^{-1}$", xy=(300, 0.06), color="m")
plt.annotate(r"$\alpha$ = 0.4 eV$^{-1}$", xy=(550, 0.02), color="y")
plt.ylabel("Reflection, $R_p$")
plt.xlabel("Time, $t$ (fs)")
# plt.gca().xaxis.set_major_locator(MultipleLocator(50))
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
plt.xlim([-150, 850])
plt.ylim([0, 0.25])

# THz
plt.subplot(4,1,4)
plt.plot(t_offset*1e15, J_THz, "k-")
plt.xlim([-150.0, 850])
plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.ylabel(r"Current, $J_{\mathrm{THz}}$ (J/m$^2$)")
plt.xlabel(r"Time, $t$ (fs)")
# ax = plt.twinx()
# plt.plot(t_offset*1e15, electric_field[:,1], "g--", zorder=-10)
# plt.ylabel("Electric field")

plt.tight_layout()
# fig.savefig("Fig3.png", dpi=300)
# %%
