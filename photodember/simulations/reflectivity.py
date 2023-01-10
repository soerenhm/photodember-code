# %%
from dataclasses import dataclass, replace, asdict
import json
from typing import Any, Callable, Dict, List, Sequence, Tuple
import numpy as np
import itertools
from pathlib import Path
from functools import partial, cached_property

from scipy.interpolate import interp1d
from scipy.special import erf

from photodember.src.constants import SI
from photodember.src.optics.helmholtz import Layer, Stack, solve_stack
from photodember.simulations.core import SimulationConfig, read_simulation_file

# Pimp my plot
# -----------------------------------------------------------------------------
# %%

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from photodember.src.transport.simulation import SimulationState

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["animation.ffmpeg_path"] = "C:/ffmpeg/bin/ffmpeg.exe"


# Utils
# -----------------------------------------------------------------------------
# %%


def select(inds, seq):
    for i in inds:
        yield seq[i]


def unzip(seq):
    return zip(*seq)


def dictproduct(d: Dict[str, Any]):
    def split_dict(pair: Tuple[str, Any]):
        k, values = pair
        try:
            return itertools.product([k], values)
        except TypeError:
            return itertools.product([k], [values])

    return map(dict, itertools.product(*map(split_dict, d.items())))


def chunked(n: int, iterable):
    iterator = iter(iterable)
    seq = (list(itertools.islice(iterator, n)) for _ in itertools.repeat(None))
    return itertools.takewhile(bool, seq)


def convolve(td, yd, f, t):
    g = f(t)
    g = g / np.sum(g)
    T = td[-1] - td[0]
    idx = np.logical_and(t >= td[0] - T / 4, t <= td[-1] + T / 4)
    z = np.convolve(g, np.interp(t[idx], td, yd))[: len(t)]
    return t - T / 4, z


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


def integrated_wavevector(E, t, tau_eph: float, samples: int = 2000):
    # Transform to uniform simulation times
    t_conv, dt_conv = np.linspace(t[0], t[-1], samples, retstep=True)
    conv = np.exp(-t_conv / tau_eph) * dt_conv
    E_int = np.array(
        [
            np.convolve(conv, np.interp(t_conv, t, E[:, i]))[: len(t_conv)]
            for i in range(0, E.shape[1])
        ]
    )
    # Transform back to the (generally) non-uniform simulation times
    E_int = np.array([np.interp(t, t_conv, E_int[i, :]) for i in range(E_int.shape[0])])
    return SI.e * E_int / SI.hbar


# Helmholtz solver
# -----------------------------------------------------------------------------
# %%
# kz = interp1d(xgrid, SI.e * efield_t_integral / SI.hbar, kind="cubic")


def create_eps_from(eps_v, n_val, om, density, mass, gam):
    def eps(x):
        nx = density(x)
        om_p = np.sqrt(SI.e**2 * nx / (SI.eps_0 * mass(x)))
        return (
            eps_v - (eps_v - 1) * (nx / n_val) - om_p**2 / (om * (om + 1j * gam(x)))
        )

    return eps


def create_stack(L, eps_v, eps_x, eps_z):
    # Doesn't make much different to add exponential tails...
    # L = 10e-6
    # eps_x = interp_with_exp_tail(xgrid, eps_x(xgrid), L, 50)
    # eps_z = interp_with_exp_tail(xgrid, eps_z(xgrid), L, 50)
    return Stack(
        [
            Layer.create(L - 1e-9, eps_o=eps_x, eps_e=eps_z),
            Layer.create(np.inf, lambda _: eps_v + 0j),
        ]
    )


def solve_refl(stack: Stack, om: float, theta: float):
    return solve_stack(stack, om, theta, "s"), solve_stack(stack, om, theta, "p")


# Load simulation and stuff
# -----------------------------------------------------------------------------
# %%


def electric_field_to_grid(efield):
    res = np.zeros_like(efield)
    res[0] = 0.5 * efield[1]
    res[1:-1] = 0.5 * (efield[1:-1] + efield[2:])
    res[-1] = 0.5 * efield[-1]
    return res


def select_nearest_simulation_states(simul_t0, simul_tfwhm, simul_t):
    t_0 = simul_t0
    t_fwhm = simul_tfwhm
    t_1 = t_0 - 2 * t_fwhm
    t_2 = t_1 + 4 * t_fwhm
    t_3 = 1e-12 + t_0
    find_nearest_t = lambda ti: np.argmin(np.abs(ti - simul_t))
    times = (
        list(np.linspace(0, t_1, 5))
        + list(np.linspace(t_1 + 1e-15, t_2, 25))
        + list(np.linspace(t_2 + 10e-15, t_3, 15))
    )
    state_idx = list(map(find_nearest_t, times))
    return state_idx


def gam_const(gam):
    return lambda x: gam * np.ones_like(x)


def gam_relmass_pow(relmass, n, gam):
    def g(x):
        return gam * relmass(x) ** n

    return g


def gam_wavevector_power(a, n, kz, gam):
    def g(x):
        v = np.maximum(1e-3 * np.ones_like(x), 1 + a * kz(x) ** 2)
        return gam * v**n
        # return gam * (1 + a * kz(x) ** 2) ** n

    return g


def electron_mass_x(m, alpha, kz):
    return lambda x: effective_mass_perp(m, alpha, kz(x))


def electron_mass_z(m, alpha, kz):
    return lambda x: effective_mass(m, alpha, kz(x))


def reduced_mass_x(mc, mv, alpha, kz):
    mx = electron_mass_x(mc, alpha, kz)
    return lambda x: 1.0 / (1.0 / mv + 1.0 / mx(x))


def reduced_mass_z(mc, mv, alpha, kz):
    mz = electron_mass_z(mc, alpha, kz)
    return lambda x: 1.0 / (1.0 / mv + 1.0 / mz(x))


MASS_EXPRESSION = [
    "electron_mass_x",
    "reduced_mass_x",
    "electron_mass_z",
    "reduced_mass_z",
    "constant",
]

GAM_EXPRESSION = ["constant", "powerfn"]


def choose_mass_function(mass_expr: str, mc, mv, alpha, kz):
    if mass_expr == "electron_mass_x":
        return electron_mass_x(mc, alpha, kz)
    elif mass_expr == "electron_mass_z":
        return electron_mass_z(mc, alpha, kz)
    elif mass_expr == "reduced_mass_x":
        return reduced_mass_x(mc, mv, alpha, kz)
    elif mass_expr == "reduced_mass_z":
        return reduced_mass_z(mc, mv, alpha, kz)
    elif mass_expr == "constant":
        return lambda x: mc * np.ones_like(x)
    else:
        raise ValueError("Unknown mass_expr.")


def choose_gam_function(gam_expr: str, gam_drude, kz):
    if gam_expr == "constant":
        return gam_const(gam_drude)
    elif gam_expr.startswith("powerfn"):
        _, a, n = gam_expr.split(" ")
        a, n = float(a), float(n)
        return gam_wavevector_power(a, n, kz, gam_drude)
        # return gam_relmass_pow(rel_mass_fn, n, gam_drude)
    else:
        raise ValueError("Unknown gam_expr.")


# Calculator
# -----------------------------------------------------------------------------
# %%


@dataclass(frozen=True)
class Calculator:
    eps_v: float
    n_val: float
    probe_wvl: float
    tau_eph: float
    alpha_eV: float
    density_scale: float
    gam_drude: float
    simul_file: str
    expr_mass_x: str
    expr_mass_z: str
    expr_gam_x: str
    expr_gam_z: str
    aoi: float = 60.0
    mc: float = 0.5
    mv: float = 3.0

    @cached_property
    def simulation_config(self):
        simulfile = self.simul_file
        metafile = simulfile.replace(".dat", ".json")
        with open(metafile, "r") as io:
            simulconf = SimulationConfig.from_dict(json.load(io))
        return simulconf

    @cached_property
    def simulation_states(self):
        simulfile = self.simul_file
        simulconf = self.simulation_config
        init_state = simulconf.initial_state
        state_t, states = read_simulation_file(simulfile, init_state)
        return state_t, states

    def create_stacks(self, simulconf, state_t, states):
        mc, mv = self.mc * SI.m_e, self.mv * SI.m_e
        x = simulconf.grid
        E = electric_field_to_grid(np.array([st.electric_field for st in states]))
        kvectors = integrated_wavevector(E, state_t, self.tau_eph)
        om = 2 * np.pi * SI.c_0 / self.probe_wvl
        alpha = self.alpha_eV / SI.e
        L = x[-1] - 0.5e-9
        eps_i = partial(create_eps_from, self.eps_v, self.n_val, om)

        def stack_from_index(index: int) -> Stack:
            state = states[index]
            kz = interp1d(x, kvectors[:, index], kind="cubic")
            mx = choose_mass_function(self.expr_mass_x, mc, mv, alpha, kz)
            mz = choose_mass_function(self.expr_mass_z, mc, mv, alpha, kz)
            gx = choose_gam_function(self.expr_gam_x, self.gam_drude, kz)
            gz = choose_gam_function(self.expr_gam_z, self.gam_drude, kz)
            n = interp1d(x, state.number_density[0] * self.density_scale)
            eps_x = eps_i(n, mx, gx)
            eps_z = eps_i(n, mz, gz)
            stack = create_stack(L, self.eps_v, eps_x, eps_z)
            return stack, {"mx": mx, "mz": mz, "gx": gx, "gz": gz}

        return stack_from_index

    def simulate(self):
        simulconf = self.simulation_config
        state_t, states = self.simulation_states

        mc, mv = self.mc * SI.m_e, self.mv * SI.m_e
        x = simulconf.grid
        E = electric_field_to_grid(np.array([st.electric_field for st in states]))
        kvectors = integrated_wavevector(E, state_t, self.tau_eph)
        om = 2 * np.pi * SI.c_0 / self.probe_wvl
        alpha = self.alpha_eV / SI.e
        L = x[-1] - 0.5e-9
        eps_i = partial(create_eps_from, self.eps_v, self.n_val, om)

        def run(index):
            state = states[index]
            kz = interp1d(x, kvectors[:, index], kind="cubic")
            mx = choose_mass_function(self.expr_mass_x, mc, mv, alpha, kz)
            mz = choose_mass_function(self.expr_mass_z, mc, mv, alpha, kz)
            gx = choose_gam_function(self.expr_gam_x, self.gam_drude, kz)
            gz = choose_gam_function(self.expr_gam_z, self.gam_drude, kz)
            n = interp1d(x, state.number_density[0] * self.density_scale)
            eps_x = eps_i(n, mx, gx)
            eps_z = eps_i(n, mz, gz)
            stack = create_stack(L, self.eps_v, eps_x, eps_z)
            Rs = solve_stack(stack, om, self.aoi, "s").R
            Rp = solve_stack(stack, om, self.aoi, "p").R
            ti = state_t[index] - simulconf.excitation_time_zero
            return ti, Rs, Rp, stack

        idx = select_nearest_simulation_states(
            simulconf.excitation_time_zero,
            simulconf.excitation_time_fwhm,
            state_t,
        )
        return map(run, idx)


def run_calculator(calc: Calculator, fn: Callable[[Stack], Any], idx: Sequence[int]):
    simulconf = calc.simulation_config
    state_t, states = calc.simulation_states

    mc, mv = calc.mc * SI.m_e, calc.mv * SI.m_e
    x = simulconf.grid
    E = electric_field_to_grid(np.array([st.electric_field for st in states]))
    kvectors = integrated_wavevector(E, state_t, calc.tau_eph)
    om = 2 * np.pi * SI.c_0 / calc.probe_wvl
    alpha = calc.alpha_eV / SI.e
    L = x[-1] - 0.5e-9
    eps_i = partial(create_eps_from, calc.eps_v, calc.n_val, om)

    def run(index):
        state = states[index]
        kz = interp1d(x, kvectors[:, index], kind="cubic")
        mx = choose_mass_function(calc.expr_mass_x, mc, mv, alpha, kz)
        mz = choose_mass_function(calc.expr_mass_z, mc, mv, alpha, kz)
        gx = choose_gam_function(calc.expr_gam_x, calc.gam_drude, kz)
        gz = choose_gam_function(calc.expr_gam_z, calc.gam_drude, kz)
        n = interp1d(x, state.number_density[0] * calc.density_scale)
        eps_x = eps_i(n, mx, gx)
        eps_z = eps_i(n, mz, gz)
        stack = create_stack(L, calc.eps_v, eps_x, eps_z)
        return fn(stack)

    return zip(select(state_t, idx), map(run, idx))


def run_and_append_to_file(calc: Calculator, save_file) -> list:
    if Path(save_file).exists():
        with open(save_file, "r") as io:
            results = json.load(io)
        calculator_settings = results.get("calculator_settings", [])
        calculator_results = results.get("calculator_results", [])
        for settings, result in zip(calculator_settings, calculator_results):
            calculator = Calculator(**settings)
            if calc == calculator:
                return result
    else:
        calculator_settings = []
        calculator_results = []
    t, Rs, Rp, _ = unzip(calc.simulate())
    result = [list(t), list(Rs), list(Rp)]
    calculator_results.append(result)
    calculator_settings.append(asdict(calc))
    with open(save_file, "w") as io:
        json.dump(
            {
                "calculator_settings": calculator_settings,
                "calculator_results": calculator_results,
            },
            io,
        )
    return result


# %%

default_settings = Calculator(
    eps_v=2.11,
    n_val=3.2e29,
    probe_wvl=700e-9,
    tau_eph=100e-15,
    alpha_eV=0.4,
    density_scale=1.0,
    gam_drude=1.2e15,
    simul_file=r"photodember\simulations\fs1_source_term_with_T_rise_1nm_125fs_muscale-0.1.dat",
    expr_mass_x="reduced_mass_x",
    expr_mass_z="reduced_mass_z",
    expr_gam_x="constant",
    expr_gam_z="constant",
    aoi=60.0,
    mc=0.5,
    mv=3.0,
)

# %%

settings = replace(
    default_settings,
    simul_file=r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_alpha-2.0.dat",
    expr_gam_z="constant",  # "powerfn 0.2e-18 3.0",
    tau_eph=200e-15,
    alpha_eV=2.0,
    # expr_gam_z="constant",
    expr_mass_z="electron_mass_z",
    expr_gam_x="constant",
    expr_mass_x="electron_mass_x",
    density_scale=1.2,
    gam_drude=1.8e15,
)
t, Rs, Rp, stack = unzip(settings.simulate())
plt.plot(t, Rs)
plt.plot(t, Rp)

# %%

plt.plot(t, Rs)
plt.plot(t, Rp)

# %%

a = 4.9e-10  # https://link.springer.com/content/pdf/10.1007/BF00552441.pdf
kBz = np.pi / a

arguments = dict(
    simul_file=[
        # r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-0.1.dat",
        # r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0.dat",
        r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_alpha-2.0.dat",
        # r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_SimpleExp-40nm.dat",
    ],
    expr_gam_x=[
        "constant",
        # "powerfn 1e-17 -0.5",
        # "powerfn 1e-18 -1.0",
        # "powerfn 1e-19 -1.5",
    ],
    expr_gam_z=[
        "constant",
        # "powerfn 1e-17 -0.5",
        # "powerfn 1e-18 -1.0",
        # "powerfn 1e-19 -1.5",
    ],
    expr_mass_x=["electron_mass_x"],
    expr_mass_z=["electron_mass_z"],
    probe_wvl=[700e-9],
    tau_eph=[150e-15, 175e-15, 200e-15],
    alpha_eV=[2.0],
    density_scale=[0.7, 0.8, 0.9, 1.0, 1.1],
    gam_drude=[
        1.1e15,
        1.2e15,
        1.3e15,
        1.4e15,
        1.5e15,
        1.6e15,
        1.7e15,
        1.8e15,
        1.9e15,
        2.0e15,
    ],
)

args = list(dictproduct(arguments))
for i, arg in enumerate(args):
    print("Progress: ", round(100 * i / len(args), 2), "%.")
    settings = replace(default_settings, **arg)
    run_and_append_to_file(settings, "scan_alot_2.json")


# %%

import time
from scipy.special import erf


with open("scan_alot_2.json", "r") as io:
    scans = json.load(io)

with open("photodember/data/delaytraces.json", "r") as io:
    exp_data = json.load(io)


def gauss_kernel(tzero, tfwhm):
    def f(t):
        x = 4 * np.log(2) * (t - tzero) / tfwhm
        return np.exp(-(x**2))

    return f


def erf_kernel(tzero, tfwhm):
    def f(t):
        x = np.sqrt(2 * np.log(2)) * (t - tzero) / tfwhm
        return 0.5 * (1 + erf(x))

    return f


results, settings = scans["calculator_results"], scans["calculator_settings"]
pred = (
    lambda s: s["probe_wvl"] == 700e-9
    and Path(s["simul_file"])
    == Path(
        r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_alpha-2.0.dat"
    )
    and s["expr_gam_z"] == "constant"
    and s["expr_gam_x"] == "constant"
    # and s["tau_eph"] == 150e-15
    and s["density_scale"] == 1.0
    # and float(s["expr_gam_x"].split(" ")[1]) < 1.0
    # and float(s["expr_gam_z"].split(" ")[2]) == 1.50
    # and s["expr_gam_z"] == "powerfn 0.5"
    # and s["expr_gam_x"] == "powerfn 0.5"
    # and s["tau_eph"] == 100e-15
)
results_, settings_ = unzip(filter(lambda x: pred(x[1]), zip(results, settings)))
batch_size = 4
for i, batch_result in enumerate(chunked(batch_size, results_)):
    plt.figure(dpi=125)
    D = exp_data["700"]
    t = np.array(D["t"]) * 1e-15
    plt.errorbar(t, D["Rs"], D["Rserr"], fmt="bo", fillstyle="none", capsize=3)
    plt.errorbar(t, D["Rp"], D["Rperr"], fmt="rs", fillstyle="none", capsize=3)
    ls = ["-", "--", "-.", ":"]
    t_kernel = np.linspace(-500e-15, 2000e-15, 2000)
    for j, r in enumerate(batch_result):
        t, Rs, Rp = r
        index = i * len(batch_result) + j
        plt.plot(
            *convolve(t, Rp, gauss_kernel(30.0e-15, 150e-15), t_kernel),
            "r",
            lw=1,
            ls=ls[j],
            label=index
        )
        plt.plot(
            *convolve(t, Rs, gauss_kernel(30.0e-15, 150e-15), t_kernel),
            "b",
            lw=1,
            ls=ls[j]
        )
    plt.xlim([t[0], t[-1]])
    plt.legend(frameon=False)
    plt.show()
    time.sleep(1.0)


# -----------------------------------------------------------------------------
# Instead of computing reflectivities, let's just compare the z-averaged permittivity\
# %%

default_settings = Calculator(
    eps_v=2.11,
    n_val=3.2e29,
    probe_wvl=700e-9,
    tau_eph=100e-15,
    alpha_eV=0.4,
    density_scale=1.0,
    gam_drude=1.2e15,
    # simul_file=r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_SimpleExp-40nm.dat",
    simul_file=r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_alpha-2.0.dat",
    # simul_file=r"photodember\simulations\fs1_source_term_with_T_rise_1nm_125fs_muscale-0.1.dat",
    expr_mass_x="reduced_mass_x",
    expr_mass_z="reduced_mass_z",
    expr_gam_x="constant",
    expr_gam_z="constant",
    aoi=60.0,
    mc=0.5,
    mv=3.0,
)

conf = default_settings.simulation_config
states_t, states = default_settings.simulation_states
states_t = np.array(states_t)
get_stack_fn = lambda settings: settings.create_stacks(conf, states_t, states)


def average_permittivity(eps_v, eps, x):
    return np.sum(eps(x) - eps_v * np.ones_like(x)) / len(x)


def av_eps(skin_depth: float, calc: Calculator, t: List[float]):
    # om = 2 * np.pi * SI.c_0 / calc.probe_wvl
    x = np.linspace(0, 10 * skin_depth, 1000)
    get_stack = get_stack_fn(calc)

    def fn(t_index: int):
        stack, pars = get_stack(t_index)
        layer = stack.first
        x_ = x[x <= layer.thickness]
        averager = np.exp(-x_ / skin_depth)
        averager /= sum(averager)
        # eps_x = np.sum(stack.first.eps_o(x_) * averager)
        # eps_z = np.sum(stack.first.eps_e(x_) * averager)
        eps_x = average_permittivity(calc.eps_v, layer.eps_o, x_)
        eps_z = average_permittivity(calc.eps_v, layer.eps_e, x_)
        mx = np.sum(pars["mx"](x_) * averager)
        mz = np.sum(pars["mz"](x_) * averager)
        gx = np.sum(pars["gx"](x_) * averager)
        gz = np.sum(pars["gz"](x_) * averager)
        return eps_x, eps_z, dict(mx=mx, mz=mz, gx=gx, gz=gz)

    idx = [np.argmin(np.abs(states_t - ti)) for ti in t]
    return map(fn, idx)


# %%

import math
from photodember.src.optics.optical_models import Drude


def _parse_tre_fit_from_dict(data, wvl, index):
    om = 2.0 * math.pi * 2.998e8 / wvl
    return TreFit(
        om=om,
        n0=data["fixed"]["n"],
        gam=data["fixed"]["Γ"],
        chi0=data["free"]["δ_re"] + 1j * data["free"]["δ_im"],
        t01=data["fixed"]["t01"],
        t02=data["free"]["t02"],
        taun=data["fixed"]["τ1"],
        taux=data["free"]["τ2"],
        eps_val=index**2.0,
    )


@dataclass(frozen=True)
class TreFit:
    om: float
    n0: float
    gam: float
    chi0: complex
    t01: float
    t02: float
    taun: float
    taux: float
    m: float = 3.92e-31
    nval: float = 3.2e29
    L: float = 40e-9
    eps_val: float = 3.0

    @staticmethod
    def load(filepath: str, wvl, index) -> "TreFit":
        with open(filepath, "r", encoding="utf8") as io:
            out = json.loads(io.read())
        return _parse_tre_fit_from_dict(out, wvl, index)


def drude_model_from_fit(fit: TreFit, t: float):
    n0 = fit.n0 * 0.5 * (1 + erf(np.sqrt(4 * np.log(2)) * ((t - fit.t01) / fit.taun)))
    n = lambda x: n0 * np.exp(-x / fit.L)
    drude = Drude(fit.eps_val, fit.nval, fit.gam, fit.m)
    return lambda x: drude.dielectric_function(fit.om, n(x))


def anisotropy_from_fit(fit: TreFit, t: float):
    chi0 = (
        fit.chi0 * 0.5 * (1 + erf(np.sqrt(4 * np.log(2)) * ((t - fit.t02) / fit.taux)))
    )
    return lambda x: chi0 * np.exp(-x / fit.L)


def permittivity_from_fit(fit: TreFit, t: float):
    eps_xx = drude_model_from_fit(fit, t)
    chi_zz = anisotropy_from_fit(fit, t)
    eps_o = eps_xx
    eps_e = lambda x: eps_xx(x) + chi_zz(x)
    return eps_o, eps_e


def average_permittivity_from_fit(fit: TreFit, t: float):
    eps_xx = drude_model_from_fit(fit, t)
    chi_zz = anisotropy_from_fit(fit, t)
    eps_o = eps_xx
    eps_e = lambda x: eps_xx(x) + chi_zz(x)
    x = np.linspace(0, fit.L * 10, 1000)
    return average_permittivity(fit.eps_val, eps_o, x), average_permittivity(
        fit.eps_val, eps_e, x
    )


with open(r"photodember\data\fit_7.json", "r") as io:
    data_fit = json.load(io)


drude_fit = TreFit.load(
    r"photodember\data\fit_7.json", 700e-9, np.sqrt(default_settings.eps_v)
)

# %%

# default_settings = Calculator(
#     eps_v=2.11,
#     n_val=3.2e29,
#     probe_wvl=700e-9,
#     tau_eph=100e-15,
#     alpha_eV=0.4,
#     density_scale=1.0,
#     gam_drude=1.2e15,
#     simul_file=r"photodember\simulations\fs1_source_term_with_T_rise_1nm_35fs_muscale-1.0_SimpleExp-40nm.dat",
#     expr_mass_x="reduced_mass_x",
#     expr_mass_z="reduced_mass_z",
#     expr_gam_x="constant",
#     expr_gam_z="constant",
#     aoi=60.0,
#     mc=0.5,
#     mv=3.0,
# )


t = np.linspace(-200e-15, 800e-15, 100)
t0 = default_settings.simulation_config.excitation_time_zero
settings = replace(
    default_settings,
    # expr_gam_z="constant",
    expr_gam_z="powerfn 1e-17 -0.5",  # "powerfn 1e-18 -1.5",
    expr_gam_x="powerfn 1e-17 -0.5",  # "powerfn 1e-19 0.5",
    expr_mass_x="electron_mass_x",
    expr_mass_z="electron_mass_z",
    tau_eph=200e-15,
    alpha_eV=2.0,
    density_scale=0.9,  # 1.9
    gam_drude=1.9e15,
)
eps_x_ref, eps_z_ref = unzip([average_permittivity_from_fit(drude_fit, ti) for ti in t])
eps_x, eps_z, pars = unzip(av_eps(40e-9, settings, t))

s, _ = get_stack_fn(settings)(70)
plt.figure()
x_ = np.linspace(0, 200e-9, 1000)
plt.plot(x_, np.real(s.first.eps_e(x_)))
plt.plot(x_, np.imag(s.first.eps_e(x_)))

plt.figure(figsize=(7, 5))
plt.subplot(2, 2, 1)
plt.plot(t, np.real(eps_x_ref), "g--", fillstyle="none")
plt.plot(t - t0, np.real(eps_x), "g-", label="re")
plt.plot(t, np.imag(eps_x_ref), "c--", fillstyle="none")
plt.plot(t - t0, np.imag(eps_x), "c-", label="im")
plt.ylabel(r"$\epsilon_x$")
plt.xlim([t[0], t[-1] - t0])

plt.legend(frameon=False)

plt.subplot(2, 2, 3)
plt.plot(t, np.real(eps_z_ref), "g--", fillstyle="none")
plt.plot(t - t0, np.real(eps_z), "g-", label="re")
plt.plot(t, np.imag(eps_z_ref), "c--", fillstyle="none")
plt.plot(t - t0, np.imag(eps_z), "c-", label="im")
plt.ylabel(r"$\epsilon_z$")
plt.xlim([t[0], t[-1] - t0])
plt.legend(frameon=False)

plt.subplot(2, 2, 2)
mx, mz, gx, gz = map(
    np.array, unzip([(p["mx"], p["mz"], p["gx"], p["gz"]) for p in pars])
)
plt.plot(t - t0, mx / mx[0], "b-", label=r"$m_x$")
plt.plot(t - t0, mz / mz[0], "r-", label=r"$m_z$")
plt.ylabel(r"$m_{\mathrm{eff}}$")
plt.xlim([t[0], t[-1] - t0])

plt.subplot(2, 2, 4)
plt.plot(t - t0, gx / gx[0], "b--")
plt.plot(t - t0, gz / gz[0], "r--")
plt.ylabel(r"$\Gamma$")
plt.ylim(ymin=0)
plt.xlim([t[0], t[-1] - t0])

plt.tight_layout()

# %%

plt.figure()

td, Rs, Rp, stack = unzip(settings.simulate())
plt.plot(td, Rs, "b-")
plt.plot(td, Rp, "r-")

# eps_x_ref, eps_z_ref = unzip([permittivity_from_fit(drude_fit, ti) for ti in td])
# stacks = [
#     Stack(
#         [
#             Layer.create(1e-6, eps_x_ref[i], eps_z_ref[i]),
#             Layer.create(math.inf, lambda x: settings.eps_v + 0j),
#         ]
#     )
#     for i in range(len(td))
# ]
# Rs2 = [solve_stack(stack, 2.35e15 * 700 / 800, 60.0, "s").R for stack in stacks]
# Rp2 = [solve_stack(stack, 2.35e15 * 700 / 800, 60.0, "p").R for stack in stacks]
# plt.plot(td, Rs2, "b--")
# plt.plot(td, Rp2, "r--")

D = exp_data["700"]
td = np.array(D["t"]) * 1e-15
plt.errorbar(td, D["Rs"], D["Rserr"], fmt="bo", fillstyle="none", capsize=3)
plt.errorbar(td, D["Rp"], D["Rperr"], fmt="rs", fillstyle="none", capsize=3)

# %%
