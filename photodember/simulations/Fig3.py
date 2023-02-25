# %%
from dataclasses import replace, asdict
import json
import numpy as np
from functools import lru_cache

from photodember.simulations.mplstyle import *
from photodember.src.constants import SI
from photodember.simulations.core import *
from photodember.src.optics.helmholtz import Stack, Layer, solve_stack


# -----------------------------------------------------------------------------
# Optical calculates for Fig. 3
# %%


default_model = OpticalModel(
    eps_v=2.11,
    n_val=3.2e29,
    probe_wvl=700e-9,
    tau_eph=120e-15,
    alpha_eV=2.0,
    density_scale=1.0,
    gam_drude=1.6e15,  # 1.6e15 is good
    simul_file=f"photodember/simulations/SiO2_fluence_tau-120fs_muh-scale-0.1.dat",
    # simul_file=r"photodember\simulations\fs_source_term_with_T_rise_1nm_35fs_mue-2e-3_muh-1e-4_MRE_alpha-2.0_2.6Jcm2.dat",
    expr_mass_x="electron_mass_x",
    expr_mass_z="electron_mass_z",
    expr_gam_x="constant",
    expr_gam_z="constant",
    aoi=60.0,
    mc=0.5,
    mv=10.0,
)


def calculate_optical_properties(x, om, th, eps_v, eps_x, eps_z):
    # Extrapolate to 6 um where eps is surely = eps_v
    L = 6e-6
    eps_x_r = right_extrapolate_with_exp_decay(
        lambda x: np.real(eps_x(x)), eps_v, x, [1.0, 1e-6]
    )
    eps_x_i = right_extrapolate_with_exp_decay(
        lambda x: np.imag(eps_x(x)), 0.0, x, [1.0, 1e-6]
    )
    eps_z_r = right_extrapolate_with_exp_decay(
        lambda x: np.real(eps_z(x)), eps_v, x, [1.0, 1e-6]
    )
    eps_z_i = right_extrapolate_with_exp_decay(
        lambda x: np.imag(eps_z(x)), 0.0, x, [1.0, 1e-6]
    )
    eps_x_extr = lambda x: eps_x_r(x) + 1j * eps_x_i(x)
    eps_z_extr = lambda x: eps_z_r(x) + 1j * eps_z_i(x)
    stack = Stack(
        [
            Layer.create(L, eps_o=eps_x_extr, eps_e=eps_z_extr),
            Layer.create(np.inf, lambda _: default_model.eps_v + 0j),
        ]
    )
    Rs = solve_stack(stack, om, th, "s").R
    Rp = solve_stack(stack, om, th, "p").R
    return dict(Rs=Rs, Rp=Rp, eps_x=eps_x_extr, eps_z=eps_z_extr)


def calculate_optical_properties_no_extrapolate(x, om, th, eps_v, eps_x, eps_z):
    # Extrapolate to 6 um where eps is surely = eps_v
    L = x[-1] - 0.1e-9
    stack = Stack(
        [
            Layer.create(L, eps_o=eps_x, eps_e=eps_z),
            Layer.create(np.inf, lambda _: default_model.eps_v + 0j),
        ]
    )
    Rs = solve_stack(stack, om, th, "s").R
    Rp = solve_stack(stack, om, th, "p").R
    return dict(Rs=Rs, Rp=Rp, eps_x=eps_x, eps_z=eps_z)


def absorption_coefficient(om, th, eps_x, eps_z):
    kappa = np.imag(np.sqrt(eps_x - eps_x * np.sin(np.deg2rad(th)) ** 2 / eps_z))
    return 2 * kappa * om / SI.c_0


conf = default_model.simulation_config
state_t, states = default_model.simulation_states
get_optical_model = default_model.create_optical_model_fn(conf, state_t, states)
om = 2 * np.pi * SI.c_0 / default_model.probe_wvl
th = 60.0
x = conf.grid

tr = np.linspace(-200e-15, 1000e-15, 100)
idx = np.unique([np.argmin(np.abs(state_t - tri)) for tri in tr])


@lru_cache
def calc(index):
    eps_x, eps_z, pars = get_optical_model(index)
    print(index)
    # result = calculate_optical_properties_no_extrapolate(
    #     x[-50:], om, th, default_model.eps_v, eps_x, eps_z
    # )
    try:
        result = calculate_optical_properties(
            x, om, th, default_model.eps_v, eps_x, eps_z
        )
    except ValueError:  # this happens when diffusion has not yet set in...
        result = calculate_optical_properties(
            x[-20:], om, th, default_model.eps_v, eps_x, eps_z
        )
    else:
        print(
            f"Warning: Failed to extrapolate at {index}. Defaulting to no extrapolate..."
        )
        result = calculate_optical_properties_no_extrapolate(
            x[-50:], om, th, default_model.eps_v, eps_x, eps_z
        )
    return dict(**result, **pars)


def gauss_kernel(tzero, tfwhm):
    def f(t):
        x = 4 * np.log(2) * (t - tzero) / tfwhm
        return np.exp(-(x**2))

    return f


# %%

t = np.array(state_t)[idx]
result = [calc(i) for i in idx]

# %%

t0 = conf.excitation_time_zero
toffset = -60.0e-15
tfwhm = 125e-15
x = np.copy(conf.grid)


# Integrated k-vector


def average_over_skindepth(om, th, x, eps_x, eps_z, array):
    """Average array over skin depth"""
    a = absorption_coefficient(om, th, eps_x, eps_z)
    a = 1 / (10e-9)
    g = np.exp(-a * x)
    g /= np.sum(g)
    return np.sum(array * g)


kz = np.array([r["kz"](x) for r in result])
kBz = np.pi / (4.9e-10)
ek = energy_vs_k(default_model.alpha_eV / SI.e, default_model.mc * SI.m_e, 0.0, 0.0, kz)

# Effective mass

mx = np.array([np.mean(r["mx"](x)[x < 20e-9]) for r in result])
mz = np.array([np.mean(r["mz"](x)[x < 20e-9]) for r in result])

# Reflectivity calculations

tconv = np.linspace(t[0] - 10 * tfwhm, t[-1] + 10 * tfwhm, 500)
Rpdata = convolve(
    t - t0, [r["Rp"] for r in result], gauss_kernel(toffset, tfwhm), tconv
)
Rsdata = convolve(
    t - t0, [r["Rs"] for r in result], gauss_kernel(toffset, tfwhm), tconv
)

# Reflectivity data

with open("photodember/data/delaytraces.json", "r") as io:
    expdata = json.load(io)["700"]


# %%

fig = plt.figure(figsize=(5, 6))

ax1 = plt.subplot(3, 1, 1)
im = plt.pcolormesh(
    (t - t0) * 1e15,
    x * 1e9,
    kz.T / kBz,
    cmap=cc.cm["gwv"],
    shading="gouraud",
    rasterized=True,
    vmax=0.35,
)
plt.xlim([-100, 800])
plt.ylim([0, 100])
ax1.yaxis.set_major_locator(MultipleLocator(25))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
plt.ylabel(r"Depth, $z$ (nm)")
plt.colorbar(
    im,
    ax=[ax1],
    location="top",
    shrink=0.75,
    label=r"Crystal momentum, $k_z / k_{\mathrm{BZ}}$",
    ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
)


ax2 = plt.subplot(3, 1, 2)
plt.plot((t - t0) * 1e15, mx / SI.m_e, "b-")
plt.plot((t - t0) * 1e15, mz / SI.m_e, "r-")
plt.ylabel(r"Eff. mass, $m / m_e$")
plt.xlim([-100, 800])
plt.ylim([0.0, 3.2])
ax2.yaxis.set_major_locator(MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
plt.annotate(r"$m_{xx}$", xy=(250, 1.05), color="b", fontsize="large")
plt.annotate(r"$m_{zz}$", xy=(90, 2.3), color="r", fontsize="large")


ax3 = plt.subplot(3, 1, 3)
plt.errorbar(
    np.array(expdata["t"]) - t0 * 1e15,
    expdata["Rs"],
    expdata["Rserr"],
    fmt="bo",
    capsize=3,
    fillstyle="none",
)
plt.plot(Rsdata[0] * 1e15, Rsdata[1], "b-")
plt.ylabel(r"Reflectivity, $R$")
plt.xlabel(r"Time, $t$ (fs)")
ax3.yaxis.set_minor_locator(MultipleLocator(0.04))

plt.errorbar(
    np.array(expdata["t"]) - t0 * 1e15,
    expdata["Rp"],
    expdata["Rperr"],
    fmt="rs",
    capsize=3,
    fillstyle="none",
)
plt.plot(Rpdata[0] * 1e15, Rpdata[1], "r-")
plt.annotate(r"$R_s$", xy=(250, 0.560), color="b", fontsize="large")
plt.annotate(r"$R_p$", xy=(150, 0.2), color="r", fontsize="large")
plt.xlim([-100, 800])
plt.ylim([0, 0.8])
# fig.tight_layout()

# fig.savefig("Fig3.png", dpi=300, facecolor="w", bbox_inches="tight")

# %% Save data

np.savez_compressed(
    "data_reflektans.npz",
    Rs_meas=np.vstack(
        [np.array(expdata["t"]) - t0 * 1e15, expdata["Rs"], expdata["Rserr"]]
    ),
    Rp_meas=np.vstack(
        [np.array(expdata["t"]) - t0 * 1e15, expdata["Rp"], expdata["Rperr"]]
    ),
    Rs_calc=np.vstack([Rsdata[0] * 1e15, Rsdata[1]]),
    Rp_calc=np.vstack([Rpdata[0] * 1e15, Rpdata[1]]),
)


# %%
