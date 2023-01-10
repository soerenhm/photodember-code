# %%
import json
import numpy as np
from functools import partial

from photodember.src.constants import SI
from photodember.simulations.core import SimulationConfig, read_simulation_file

# Pimp my plot
# -----------------------------------------------------------------------------
# %%

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["animation.ffmpeg_path"] = "C:/ffmpeg/bin/ffmpeg.exe"

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


def electric_field_to_grid(efield):
    res = np.zeros_like(efield)
    res[0] = 0.5 * efield[1]
    res[1:-1] = 0.5 * (efield[1:-1] + efield[2:])
    res[-1] = 0.5 * efield[-1]
    return res


x = simulconf.grid
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


# Semi-classical equation of motion
# -----------------------------------------------------------------------------
# %% Semi-classical equation of motion

dt = t[1] - t[0]
# E = np.array([simul.ambipolar_electric_field(st) for st in states])
k = -SI.e / SI.hbar * np.cumsum(electric_field * dt, axis=0)

a = 4.9e-10  # https://link.springer.com/content/pdf/10.1007/BF00552441.pdf
kBz = np.pi / a

x0 = 0.0  # nm
ix = np.argmin(np.abs(x - x0))

plt.figure(figsize=(5, 4), dpi=150)
plt.plot(x_nm, -k[400, :] / kBz, "g-", label="t = 0.2 ps")
plt.plot(x_nm, -k[990, :] / kBz, "c-", label="t = 0.6 ps")
plt.plot(x_nm, -k[-1, :] / kBz, "m-", label="t = 1.0 ps")
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
    q = SI.hbar * k / np.sqrt(m)
    return (-1 + np.sqrt(1 + 2 * alpha * q**2)) / (2 * alpha)


def effective_mass(m, alpha, kx):
    q = SI.hbar * kx / np.sqrt(m)
    return m * (1 + 2 * alpha * q**2) ** (3 / 2)


def effective_mass_perp(m, alpha, kx):
    q = SI.hbar * kx / np.sqrt(m)
    return m * (1 + 2 * alpha * q**2) ** 0.5


# Convolve on a uniform temporal grid
tau_eph = 100e-15
t_conv, dt_conv = np.linspace(t[0], t[-1], 2000, retstep=True)
conv = np.exp(-t_conv / tau_eph) * dt_conv
electric_field_t_integral = np.array(
    [
        np.convolve(conv, np.interp(t_conv, t, electric_field[:, i]))[: len(t_conv)]
        for i in range(0, electric_field.shape[1])
    ]
)
# Transform back to the (generally) non-uniform simulation times
electric_field_t_integral = np.array(
    [
        np.interp(t, t_conv, electric_field_t_integral[i, :])
        for i in range(electric_field_t_integral.shape[0])
    ]
)


# %%

esurf = electric_field[:, 1]
tt = np.copy(t_conv)  # np.linspace(0, 3e-12, 2000)
esurff = np.interp(tt, t, esurf)
c = np.exp(-tt / tau_eph) * np.gradient(tt)
# c /= np.sum(c)  # * (tt[1] - tt[0])
plt.plot(tt, np.convolve(c, esurff)[: len(tt)])
plt.plot(t, electric_field_t_integral[1, :])
plt.plot(tt, np.cumsum(esurff) * dt_conv)


# THz 1
# -----------------------------------------------------------------------------
# %%
from scipy.signal import savgol_filter

skindepth_THz = 100e-9
dx = x[1] - x[0]
THz_averager = np.repeat(
    np.expand_dims(np.exp(-(x + 0.5 * dx) / skindepth_THz) * dx, 0), len(t), axis=0
)
J_tot = np.array([sum(simul.charge_current_density(state)) for state in states])
J_THz = np.sum(J_tot * THz_averager, axis=1)
dJdt = np.gradient(J_THz, t, axis=0)
d2Jdt2 = np.gradient(dJdt, t, axis=0)
Ens = SI.mu_0 / (6 * np.pi * SI.c_0) * savgol_filter(d2Jdt2, 10, 1)
Prad = -Ens * J_THz

plt.plot(t_fs, Prad)
plt.xlim([0, 1500])


# Helmholtz....
# -----------------------------------------------------------------------------
# %%

from scipy.interpolate import interp1d
from scipy.optimize import minimize
from photodember.src.optics.helmholtz import solve_stack, Stack, Layer

# Optical properties of fused silica
eps_v = 2.118
n_val = 3.2e29
mc = 0.5 * SI.m_e
mv = 3.0 * SI.m_e
probe_om = 800 / 700 * 2.35e15
alpha = 0.2 / SI.e
xgrid = np.copy(x)
dx = xgrid[1] - xgrid[0]


def create_eps_from(om, density, mass, gam):
    def eps(x):
        nx = density(x)
        om_p = np.sqrt(SI.e**2 * nx / (SI.eps_0 * mass(x)))
        return (
            eps_v - (eps_v - 1) * (nx / n_val) - om_p**2 / (om * (om + 1j * gam(x)))
        )

    return eps


# The electron density isn't zero at the edges (takes way too long to simulate),
# so we fit an exponentially decaying function to the dielectric function in space
# until we reach the unpertubed material properties.


def exp_decay(fv, p, x):
    return fv + p[0] * np.exp(-x / p[1])


def fit_exp_decay(fv, x, y, p0=[-1.0, 1e-6]):
    def f(p):
        return exp_decay(fv, p, x)

    def ll(p):
        return np.sum((y - f(p)) ** 2)

    return minimize(ll, p0, method="Nelder-Mead")


def interp_with_exp_tail(x, eps, xfinal, fit_last_n_pts: int = 50):
    n = fit_last_n_pts
    p1 = fit_exp_decay(eps_v, x[-n:], np.real(eps[-n:]), [-1, 1e-6]).x
    p2 = fit_exp_decay(0.0, x[-n:], np.imag(eps[-n:]), [1, 1e-6]).x
    f1 = partial(exp_decay, eps_v, p1)
    f2 = partial(exp_decay, 0.0, p2)
    # sample new eps...
    xs = [x[:-n], x[-n] + np.linspace(0.0, xfinal, len(x))]
    eps_samples = [eps[:-n], f1(xs[1]) + 1j * f2(xs[1])]
    return interp1d(np.append(*xs), np.append(*eps_samples), kind="cubic")


def create_stacks(alpha, gam, state, efield_t_integral):
    k_z = interp1d(xgrid, SI.e * efield_t_integral / SI.hbar, kind="cubic")
    mx = lambda x: effective_mass_perp(mv, alpha, k_z(x))  # 1.0 / (
    #     1.0 / effective_mass_perp(mv, alpha, k_z(x))
    #     + 1.0 / effective_mass_perp(mc, alpha, k_z(x))
    # )
    mz = lambda x: effective_mass(mv, alpha, k_z(x))  # 1.0 / (
    #     1.0 / effective_mass(mv, alpha, k_z(x))
    #     + 1.0 / effective_mass(mc, alpha, k_z(x))
    # )
    gx = lambda x: gam  # * (mc/mx(x))**(1.5)
    gz = lambda x: gam  # * (mc/mz(x))**(1.5)
    SCALE_DENSITY = 1.0
    L = x[-1] - 1e-9
    eps_x = create_eps_from(
        probe_om, interp1d(xgrid, state.number_density * SCALE_DENSITY), mx, gx
    )
    eps_z = create_eps_from(
        probe_om, interp1d(xgrid, state.number_density * SCALE_DENSITY), mz, gz
    )
    # Doesn't make much different to add exponential tails...
    L = 10e-6
    eps_x = interp_with_exp_tail(xgrid, eps_x(xgrid), L, 50)
    eps_z = interp_with_exp_tail(xgrid, eps_z(xgrid), L, 50)
    stack_s = Stack(
        [
            Layer.create(L - 1e-9, eps_x),
            Layer.create(np.inf, lambda _: eps_v + 0j),
        ]
    )
    stack_p = Stack(
        [
            Layer.create(L - 1e-9, eps_o=eps_x, eps_e=eps_z),
            Layer.create(np.inf, lambda _: eps_v + 0j),
        ]
    )
    return stack_s, stack_p


def solve_stacks(om, aoi, stacks):
    ss = solve_stack(stacks[0], om, aoi, "s")
    sp = solve_stack(stacks[1], om, aoi, "p")
    return ss, sp


def solve_helmholtz(alpha, gam, state, efield_t_integral):
    return solve_stacks(
        probe_om, 60.0, create_stacks(alpha, gam, state, efield_t_integral)
    )


# create_stacks(0.1/SI.e, lambda _)


# Solve Helmholtz....
# -----------------------------------------------------------------------------
# %%

t_0 = simulconf.excitation_time_zero
t_fwhm = simulconf.excitation_time_fwhm
t_1 = t_0 - 2 * t_fwhm
t_2 = t_1 + 4 * t_fwhm
t_3 = 1e-12 + simulconf.excitation_time_zero
find_nearest_t = lambda ti: np.argmin(np.abs(ti - t))
times_Rp = (
    list(np.linspace(0, t_1, 5))
    + list(np.linspace(t_1 + 1e-15, t_2, 25))
    + list(np.linspace(t_2 + 10e-15, t_3, 15))
)
state_inds = list(map(find_nearest_t, times_Rp))

alphas_int = [0.1, 0.2, 0.4]
alphas = np.array(alphas_int) / SI.e
gam = 1.8e15
hhsols = [
    [
        solve_helmholtz(alpha, gam, electron_states[i], electric_field_t_integral[:, i])
        for i in state_inds
    ]
    for alpha in alphas
]

# FIGURE 3
# -----------------------------------------------------------------------------
# %%

ky = 0.0
kz = 0.0
kx = np.linspace(-0.8, 0.8, 200) * kBz
mc = 0.5 * SI.m_e
# ek = energy_vs_k(alpha, mc, kx, ky, kz)

delta_k = SI.e * electric_field_t_integral / SI.hbar / kBz
Ne = np.array([st.number_density for st in electron_states]).T
skindepth = 40  # nm; good until about 40 nm...
p = np.exp(-x_nm / skindepth)
p /= np.sum(p)
P = np.repeat(np.expand_dims(p, 1), len(t), axis=1)
t_offset = np.array(t) - simulconf.excitation_time_zero  # - t[peak_density_tzero]


fig = plt.figure(figsize=(4.5, 6.0), dpi=125)
plt.subplot(3, 1, 1)
kxt = np.sum(delta_k * P, axis=0)
plt.plot(np.array(t_offset) * 1e15, kxt, "k-")
plt.xlim([-100, 1000])
plt.ylim([0, 0.45])
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))
plt.gca().xaxis.set_minor_locator(MultipleLocator(40))
plt.gca().xaxis.set_major_locator(MultipleLocator(200))
plt.ylabel(r"Momentum, $\Delta k / k_{\mathrm{BZ}}$")
plt.xlabel(r"Time, $t$ (fs)")

plt.subplot(3, 1, 2)
ls = ["c-", "m--", "y-."]
for i, alpha in enumerate(alphas_int):
    mxt = np.interp(kxt * kBz, kx, effective_mass(mc, alpha / SI.e, kx)) / mc
    # mx = effective_mass(alpha/SI.e, kx) / mc
    plt.plot(np.array(t_offset) * 1e15, mxt, ls[i])
plt.xlabel("")
plt.xlim([-100, 1000])
plt.ylim([1, 2.8])
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(40))
plt.gca().xaxis.set_major_locator(MultipleLocator(200))
plt.annotate(r"$\alpha$ = 0.1 eV$^{-1}$", xy=(150, 1.4), color="c")
plt.annotate(r"$\alpha$ = 0.2 eV$^{-1}$", xy=(200, 1.8), color="m")
plt.annotate(r"$\alpha$ = 0.4 eV$^{-1}$", xy=(320, 2.5), color="y")
plt.xlabel(r"Time, $t$ (fs)")
plt.ylabel(r"Effective mass, $m_e / m_c$")

Rps = []
Rss = []
for hhsola in hhsols:
    Rps.append([elem[1].R for elem in hhsola])
    Rss.append([elem[0].R for elem in hhsola])

plt.subplot(3, 1, 3)
# Reflection, transmission
plt.plot(t_offset[state_inds] * 1e15, Rps[0], "c-")
plt.plot(t_offset[state_inds] * 1e15, Rps[1], "m--")
plt.plot(t_offset[state_inds] * 1e15, Rps[2], "y-.")
plt.annotate(r"$\alpha$ = 0.1 eV$^{-1}$", xy=(250, 0.13), color="c")
plt.annotate(r"$\alpha$ = 0.2 eV$^{-1}$", xy=(300, 0.06), color="m")
plt.annotate(r"$\alpha$ = 0.4 eV$^{-1}$", xy=(550, 0.02), color="y")
plt.ylabel("Reflection, $R_p$")
plt.xlabel("Time, $t$ (fs)")
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.02))
plt.gca().xaxis.set_minor_locator(MultipleLocator(40))
plt.gca().xaxis.set_major_locator(MultipleLocator(200))
plt.xlim([-100, 1000])
plt.ylim([0, 0.35])
# plt.ylim([0, 0.25])

# THz
# plt.subplot(4, 1, 4)
# plt.plot(t_offset * 1e15, J_THz, "k-")
# plt.plot(t_offset * 1e15, savgol_filter(Ens, 30, 3), "k-")
# plt.xlim([-100.0, 1000])
# plt.gca().xaxis.set_major_locator(MultipleLocator(100))
# plt.ylabel(r"Current, $J_{\mathrm{THz}}$ (J/m$^2$)")
# plt.xlabel(r"Time, $t$ (fs)")
# ax = plt.twinx()
# plt.plot(t_offset*1e15, electric_field[:,1], "g--", zorder=-10)
# plt.ylabel("Electric field")

plt.tight_layout()
fig.savefig("Fig3.png", dpi=300)


# THz radiation
# -----------------------------------------------------------------------------
# %%

from scipy.fft import fft, fftfreq, fftshift

dJdt2 = np.gradient(savgol_filter(J_THz, 10, 1), t)
t2 = np.linspace(t[-1], 20e-12, 2000)
# plt.plot(t, dJdt2)

opt = fit_exp_decay(
    0.0,
    t[-100:],
    dJdt2[-100:],
    [1e15, 1e-11],
)
dJdt_tail = partial(exp_decay, 0.0, opt.x)
dJdt_full = interp1d(np.append(t[:-100], t2), np.append(dJdt2[:-100], dJdt_tail(t2)))

# plt.plot(t3, dJdt_full(t3))

t3 = np.linspace(0, 19e-12, 10000)
dJdt_f = fftshift(fft(dJdt_full(t3)))
f = fftshift(fftfreq(len(t3), t3[1] - t3[0]))
f_THz = f * 1e-12

om_THz = f * 2 * np.pi
om = om_THz[len(t3) // 2 + np.argmax(np.abs(dJdt_f[: len(t3) // 2]))]
eps_THz = create_eps_from(om, lambda x: 1e28, lambda x: mc, lambda x: 1e15)(0)
n_THz = np.sqrt(eps_THz)


def th_in_from(n_THz, th_out):
    return np.arcsin(np.sin(th_out) / np.real(n_THz))


def Erad_out(th_out, om, n_THz):
    k0 = om / SI.c_0
    th_in = th_in_from(n_THz, th_out)
    kx = k0 * np.sin(th_out)
    kz2 = k0 * np.cos(th_out)
    kz1 = np.sqrt((n_THz * k0) ** 2 - kx**2)
    I = np.imag(kz1) < 0
    kz1[I] = -kz1[I]
    tp = 2 * kz1 / (kz1 + n_THz**2 * kz2) * n_THz
    Tp = np.cos(th_out) / (np.real(n_THz) * np.cos(th_in)) * np.abs(tp) ** 2
    return Tp * np.sin(th_in) ** 2.0
    # return tp * np.sin(th_in)


th = np.linspace(-np.pi / 2, np.pi / 2, 200)
th_in = th_in_from(n_THz, th)
b = Erad_out(th, om, n_THz)

# fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
# ax.plot(th_in, b)

plt.figure(figsize=(4.5, 5))
plt.subplot(2, 1, 1)
t3 = np.append(t, t2)
plt.plot(t3 * 1e12, dJdt_full(t3), "b-")
plt.ylabel(r"d$J$/d$t$")
plt.xlabel("Time, $t$ (ps)")
plt.xlim([0.0, 0.5])

plt.subplot(2, 2, 3)
plt.plot(f_THz, np.abs(dJdt_f / dJdt_f.min()) ** 2, "b-")
plt.xlim([0, 40])
plt.ylabel(f"PSD (arb. units)")
plt.xlabel(f"Frequency, $f$ (THz)")
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

ax3 = plt.subplot(2, 2, 4, projection="polar")
plt.plot(th_in, b, "b-")
ax3.set_thetamin(-90)
ax3.set_thetamax(90)

plt.tight_layout()
plt.savefig("Fig4.png", dpi=300, facecolor="w")

# %%

# om_THz = f * 2 * np.pi
# om = om_THz[len(t3) // 2 + np.argmax(np.abs(dJdt_f[: len(t3) // 2]))]
# eps_THz = create_eps_from(om, lambda x: 1e28, lambda x: mc, lambda x: 1e15)(0)
# n_THz = np.sqrt(eps_THz)


# def th_in_from(n_THz, th_out):
#     return np.arcsin(np.sin(th_out) / np.real(n_THz))


# def Erad_out(th_out, om, n_THz):
#     k0 = om / SI.c_0
#     th_in = th_in_from(n_THz, th_out)
#     kx = k0 * np.sin(th_out)
#     kz2 = k0 * np.cos(th_out)
#     kz1 = np.sqrt((n_THz * k0) ** 2 - kx**2)
#     I = np.imag(kz1) < 0
#     kz1[I] = -kz1[I]
#     tp = 2 * kz1 / (kz1 + n_THz**2 * kz2) * n_THz
#     Tp = np.cos(th_out) / (np.real(n_THz) * np.cos(th_in)) * np.abs(tp) ** 2
#     return Tp * np.sin(th_in) ** 2.0
#     # return tp * np.sin(th_in)


# th = np.linspace(-np.pi / 2, np.pi / 2, 200)
# th_in = th_in_from(n_THz, th)
# b = Erad_out(th, om, n_THz)
# plt.plot(np.rad2deg(th_in), b)

# fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
# ax.plot(th_in, b)
# ax.set_rmax(2)
# ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
# ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
# ax.grid(True)

# %%

with open("photodember/data/delaytraces.json") as fstream:
    tre_data = json.load(fstream)

# %%

tre_t = np.array(tre_data["700"]["t"]) - 150

plt.figure()
plt.errorbar(
    tre_t,
    tre_data["700"]["Rp"],
    tre_data["700"]["Rperr"],
    fmt="ro",
    fillstyle="none",
    capsize=3,
)
plt.plot(t_offset[state_inds] * 1e15, Rps[1], "r-")
plt.xlim([-150, 800])
plt.ylim([0, 0.3])
ax = plt.twinx()
plt.errorbar(
    tre_t,
    tre_data["700"]["Rs"],
    tre_data["700"]["Rserr"],
    fmt="bs",
    fillstyle="none",
    capsize=3,
)
plt.plot(t_offset[state_inds] * 1e15, Rss[1], "b-")
plt.ylim([0, 0.8])

# %%

i = np.argmax(Ne[0, :]) + 100
stacks = create_stacks(0.8, 1.5e15, electron_states[i], electric_field_t_integral[:, i])

plt.plot(x, np.real(stacks[0].first.eps_o(x)))
plt.plot(x, np.imag(stacks[0].first.eps_o(x)))
# %%


from functools import partial

eps = stacks[0].first.eps_o(x)
data = (x, np.real(eps), np.imag(eps))


opt = fit_exp_decay(eps_v, data[0][-50:], data[1][-50:], [-1, 1e-6])
f1 = partial(exp_decay, eps_v, np.copy(opt.x))

opt = fit_exp_decay(0.0, data[0][-50:], data[2][-50:], [1.0, 1e-6])
f2 = partial(exp_decay, 0.0, opt.x)

xx = np.linspace(1e-6, 10e-6, 100)
# plt.plot(data[0][-100:], data[1][-100:])
# plt.plot(xx, f1(xx))

plt.plot(data[0][-50:], data[2][-50:])
plt.plot(xx, f2(xx))

# %%


# %%

eps2 = add_exp_tail(x, eps, 10e-6, 50)
xx = np.linspace(0, 10e-6, 1000)
plt.plot(xx, eps2(xx).real)
plt.plot(xx, eps2(xx).imag)
# %%
