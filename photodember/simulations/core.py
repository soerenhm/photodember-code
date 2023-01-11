from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import partial, cached_property
import gc
import pathlib
import itertools
import json
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import erf

from photodember.src.constants import SI
from photodember.src.transport.fdtable import FermiDiracTable
from photodember.src.transport.fdinterp import interpolate_particle_attributes
from photodember.src.transport.core import ParticleAttributes
from photodember.src.transport.simulation import *
from photodember.src.optics.helmholtz import Layer, Stack, solve_stack


# Utils
# -----------------------------------------------------------------------------


def select(inds, seq):
    """Select indices from a sequence."""
    for i in inds:
        yield seq[i]


def unzip(seq):
    return zip(*seq)


def dictproduct(d: Dict[str, Any]):
    """Sequence of dictionaries containing all possible combination of values."""

    def split_dict(pair: Tuple[str, Any]):
        k, values = pair
        try:
            return itertools.product([k], values)
        except TypeError:
            return itertools.product([k], [values])

    return map(dict, itertools.product(*map(split_dict, d.items())))


def chunked(n: int, iterable):
    """Takes in iterable and divides it into chunks of size `n`."""
    iterator = iter(iterable)
    seq = (list(itertools.islice(iterator, n)) for _ in itertools.repeat(None))
    return itertools.takewhile(bool, seq)


def convolve(td, yd, f, t):
    """Convolution between data `(td, yd)` and `f` evaluated at `t`."""
    g = f(t)
    g = g / np.sum(g)
    T = td[-1] - td[0]
    idx = np.logical_and(t >= td[0] - T / 4, t <= td[-1] + T / 4)
    z = np.convolve(g, np.interp(t[idx], td, yd))[: len(t)]
    return t - T / 4, z


def exp_decay(fv, p, x):
    return fv + p[0] * np.exp(-x / p[1])


def fit_exp_decay(fv, x, y, p0=[-1.0, 1e-6]):
    def f(p):
        return exp_decay(fv, p, x)

    def ll(p):
        return np.sum((y - f(p)) ** 2)

    return minimize(ll, p0, method="Nelder-Mead", tol=1e-12)


def right_extrapolate_with_exp_decay(func, final_value, x, p0):
    opt = fit_exp_decay(final_value, x, func(x), p0=p0)
    xmax = min(x)

    if not opt.success:
        if p0[1] > 1e4:
            raise ValueError("Could not fit data")
        else:
            return right_extrapolate_with_exp_decay(
                func, final_value, x, [p0[0], p0[1] * 1e3]
            )
        # print("Could not fit data. Perhaps because it's a constant function?")
    p = opt.x

    def f(t):
        if np.ndim(t) == 0:
            if t < xmax:
                return func(t)
            else:
                return exp_decay(final_value, p, t)
        else:
            t = np.asarray(t)
            y = np.zeros_like(t)
            I = t < xmax
            y[I] = func(t[I])
            y[~I] = exp_decay(final_value, p, t[~I])
            return y

    return f


def electric_field_to_grid(efield: ArrayF64) -> ArrayF64:
    """Electric field evaluated on grid points.

    In the transport simulations, the electric field is saved halfway between
    grid points. This function maps that electric field onto the grid points."""
    res = np.zeros_like(efield)
    res[0] = 0.5 * efield[1]
    res[1:-1] = 0.5 * (efield[1:-1] + efield[2:])
    res[-1] = 0.5 * efield[-1]
    return res


# -----------------------------------------------------------------------------
# Simulation functions


def advance(
    pde: OdeFunction, init_state: SimulationState, ti: float, dt: float
) -> SimulationState:
    gc.collect()  # sometimes memory consumption keeps rising, dunno why... this helps...
    return solve_pde(pde, init_state, [ti, ti + dt])[-1]


def save_state_in(f, state: SimulationState, t: float):
    s = np.append([t], state.array).astype(np.float64).tobytes()
    f.write(s)


def simulation_step(
    f, pde: OdeFunction, init_state: SimulationState, ti: float, tf: float
):
    new_state = advance(pde, init_state, ti, tf - ti)
    save_state_in(f, new_state, tf)
    return new_state


def read_simulation_file(
    filename: str, initial_state: SimulationState
) -> Tuple[List[float], List[SimulationState]]:
    t = []
    states = []
    chunksize = len(initial_state.array) + 1
    with open(filename, "rb") as io:
        content = np.frombuffer(io.read(), dtype=np.float64)
        start, stop = 0, chunksize
        while stop <= len(content):
            section = content[start:stop]
            t.append(section[0])
            state = SimulationState(
                section[1:], initial_state.grid_points, initial_state.number_particles
            )
            states.append(state)
            start, stop = stop, stop + chunksize
    return t, states


def run_simulation(
    pde: OdeFunction, save_file: str, init_state: SimulationState, times: ArrayLike
):
    times = np.atleast_1d(times)
    t = list(times)
    ti = t.pop(0)
    # Scan file for existing entries
    if pathlib.Path(save_file).exists():
        read_t, read_states = read_simulation_file(save_file, init_state)
        if len(read_t) > 0:
            if ti == read_t[0] and all(
                np.isclose(init_state.array, read_states[0].array)
            ):
                ti = read_t[-1]
                init_state = read_states[-1]
                t = list(filter(lambda t: t > ti, times))
            else:
                raise ValueError(
                    "Simulation file already exists but saved states differ"
                )
    else:
        with open(save_file, "wb") as f:
            save_state_in(f, init_state, ti)
    # Loop
    states = [init_state]
    while len(t) > 0:
        tf = t.pop(0)
        print(f"Advancing state from t={ti} to t={tf}...")
        with open(save_file, "ab") as f:
            new_state = simulation_step(f, pde, init_state, ti, tf)
        states.append(new_state)
        ti = tf
        init_state = new_state
    return states


# -----------------------------------------------------------------------------
# Excitation data


@dataclass
class ExcitationData:
    position: NDArray
    number_density: NDArray
    energy_per_particle: NDArray


def load_excitation_data(datafile: str) -> ExcitationData:
    data = np.loadtxt(datafile)
    position = data[:, 0] * 1e-6
    number_density = data[:, 1]
    energy_per_particle = data[:, 2] * SI.e
    return ExcitationData(position, number_density, energy_per_particle)


def to_simulation_grid(
    data: ExcitationData, grid_coordinates: NDArray
) -> ExcitationData:
    number_density = sp.interpolate.interp1d(
        data.position, data.number_density, kind="cubic"
    )(grid_coordinates)
    energy_per_particle = sp.interpolate.interp1d(
        data.position, data.energy_per_particle, kind="cubic"
    )(grid_coordinates)
    return ExcitationData(grid_coordinates, number_density, energy_per_particle)


def particle_temperatures(
    energy_per_particle: NDArray, particle_masses: List[float]
) -> List[NDArray]:
    inverse_masses = np.divide(1.0, particle_masses)
    temperature = energy_per_particle * (2 / 3) / SI.kB

    def particle_temperature(inverse_mass: float) -> NDArray:
        return temperature * (inverse_mass / np.sum(inverse_masses))

    return [particle_temperature(inv_mass) for inv_mass in inverse_masses]


# -----------------------------------------------------------------------------
# Particle data


@dataclass
class ParticleData:
    charge: float
    mobility: float
    source: str

    @property
    def mass(self) -> float:
        return (
            FermiDiracTable.load_meta(self.source.replace(".csv", ".json"))[
                "relativeMass"
            ]
            * SI.m_e
        )


def set_mobility(
    table: FermiDiracTable, meta: dict, new_value: float
) -> Tuple[FermiDiracTable, dict]:
    relaxtime = meta["relaxationTime"]
    old_value = SI.e * relaxtime / (meta["relativeMass"] * SI.m_e)
    scale = new_value / old_value
    new_table = table.scale_mobility(scale)
    new_meta = dict(meta)
    new_meta["relaxationTime"] = meta["relaxationTime"] * scale
    return new_table, new_meta


def load_particle_attributes(data: ParticleData) -> ParticleAttributes:
    table = FermiDiracTable.load_data(data.source)
    meta = FermiDiracTable.load_meta(data.source.replace(".csv", ".json"))
    new_table, _ = set_mobility(table, meta, data.mobility)
    return interpolate_particle_attributes(new_table, data.charge, SI.kB)


# -----------------------------------------------------------------------------
# Fused silica simulation


def heat_capacity_from_state(attrs: ParticleAttributes, state: ParticleState):
    return heat_capacity(
        np.zeros(state.grid_points),
        attrs.number_density,
        attrs.energy_density,
        state.temperature,
        state.red_chemical_potential,
    )


# # Legacy source term... remove?
# def create_source_term_legacy(N: int, particles: List[ParticleAttributes], final_density: NDArray, tfwhm: float, t0: float, geh: float, gph: float, Tph: float) -> OdeFunction:
#     assert len(particles) == 2
#     indexer = StateIndexer(N, 2)
#     elec_attrs, hole_attrs = particles
#     elec = indexer.particle(0)
#     hole = indexer.particle(1)
#     out = np.zeros(indexer.length)

#     A = np.sqrt(4*np.log(2)/np.pi)/tfwhm
#     G = np.zeros(indexer.grid_points)
#     def d_dt(t):
#         G[:] = final_density * A * np.exp(-4*np.log(2)*((t-t0)/tfwhm)**2)
#         return G

#     def rise(t):
#         return .5*(1 + erf(np.sqrt(4*np.log(2))*(t-t0)/tfwhm))

#     def S(t, y):
#         out.fill(0.0)
#         G = d_dt(t)
#         out[elec.number_density] += G
#         out[hole.number_density] += G

#         Te, etae = y[elec.temperature], y[elec.red_chemical_potential]
#         Th, etah = y[hole.temperature], y[hole.red_chemical_potential]

#         dUe_dNe = elec_attrs.energy_density.deta(Te, etae) / elec_attrs.number_density.deta(Te, etae) # type: ignore
#         dUh_dNh = hole_attrs.energy_density.deta(Th, etah) / hole_attrs.number_density.deta(Th, etah) # type: ignore

#         # if t < t0 + 1.5*tfwhm:
#         #     out[elec.energy_density] += G*dUe_dNe #- geh*(Te - Th) - gph*(Te - Tph)
#         #     out[hole.energy_density] += G*dUh_dNh #- geh*(Th - Te) - gph*(Th - Tph)
#         # else:
#         r = rise(t)
#         out[elec.energy_density] += G*dUe_dNe - r*(geh*(Te - Th) + gph*(Te - Tph))
#         out[hole.energy_density] += G*dUh_dNh - r*(geh*(Th - Te) + gph*(Th - Tph))
#         return out
#     return S


def create_source_term_T_rise(
    N: int,
    particles: List[ParticleAttributes],
    final_density: NDArray,
    final_temperatures: List[NDArray],
    tfwhm: float,
    t0: float,
    geh: float,
    gph: float,
    Tph: float,
) -> OdeFunction:
    assert len(particles) == 2
    indexer = StateIndexer(N, 2)
    elec_attrs, hole_attrs = particles
    elec = indexer.particle(0)
    hole = indexer.particle(1)
    out = np.zeros(indexer.length)
    G, dTe, dTh, He, Hh = [np.zeros(N) for _ in range(5)]

    A = np.sqrt(4 * np.log(2) / np.pi) / tfwhm

    def G_(t):
        G[:] = final_density * A * np.exp(-4 * np.log(2) * ((t - t0) / tfwhm) ** 2)
        return G

    def dTeh(t):
        dTe[:] = (
            (final_temperatures[0] - Tph)
            * A
            * np.exp(-4 * np.log(2) * ((t - t0) / tfwhm) ** 2)
        )
        dTh[:] = (
            (final_temperatures[1] - Tph)
            * A
            * np.exp(-4 * np.log(2) * ((t - t0) / tfwhm) ** 2)
        )
        return dTe, dTh

    def rise(t):
        return 0.5 * (1 + erf(np.sqrt(4 * np.log(2)) * (t - t0) / tfwhm))

    def S(t, y):
        out.fill(0.0)

        G = G_(t)
        dTe, dTh = dTeh(t)
        out[elec.number_density] += G
        out[hole.number_density] += G

        # Electron
        Te, etae = y[elec.temperature], y[elec.red_chemical_potential]
        Ue, Ne = elec_attrs.energy_density, elec_attrs.number_density
        He[:] = (
            Ue.dT(Te, etae) - Ue.deta(Te, etae) * Ne.dT(Te, etae) / Ne.deta(Te, etae)
        ) * dTe + Ue.deta(Te, etae) / Ne.deta(Te, etae) * G

        # Hole
        Th, etah = y[hole.temperature], y[hole.red_chemical_potential]
        Uh, Nh = hole_attrs.energy_density, hole_attrs.number_density
        Hh[:] = (
            Uh.dT(Th, etah) - Uh.deta(Th, etah) * Nh.dT(Th, etah) / Nh.deta(Th, etah)
        ) * dTh + Uh.deta(Th, etah) / Nh.deta(Th, etah) * G

        r = rise(t)
        out[elec.energy_density] += He - r * (geh * (Te - Th) + gph * (Te - Tph))
        out[hole.energy_density] += Hh - r * (geh * (Th - Te) + gph * (Th - Tph))
        return out

    return S


@dataclass(frozen=True)
class SimulationConfig:
    domain_size: float
    resolution: float
    initial_electron_hole_density: float
    excitation_datafile: str
    excitation_time_fwhm: float
    excitation_time_zero: float
    fdtable_conduction_band: str
    fdtable_valence_band: str
    electron_mobility: float
    hole_mobility: float
    electron_hole_thermalization_time: float
    carrier_phonon_thermalization_time: float
    phonon_temperature: float
    relative_permittivity: float

    @cached_property
    def grid(self) -> NDArray:
        return np.arange(0, self.domain_size, step=self.resolution)

    @property
    def grid_points(self) -> int:
        return len(self.grid)

    @property
    def electron_data(self) -> ParticleData:
        return ParticleData(-SI.e, self.electron_mobility, self.fdtable_conduction_band)

    @property
    def hole_data(self) -> ParticleData:
        return ParticleData(SI.e, self.hole_mobility, self.fdtable_valence_band)

    @cached_property
    def particle_attributes(self):
        return [
            load_particle_attributes(self.electron_data),
            load_particle_attributes(self.hole_data),
        ]

    @cached_property
    def excitation_data(self) -> ExcitationData:
        return to_simulation_grid(
            load_excitation_data(self.excitation_datafile), self.grid
        )

    @property
    def particle_masses(self) -> List[float]:
        return [self.electron_data.mass, self.hole_data.mass]

    @property
    def initial_temperatures(self) -> List[NDArray]:
        Tph = self.phonon_temperature
        N = self.grid_points
        return [Tph * np.ones(N), Tph * np.ones(N)]

    @property
    def excitation_temperatures(self) -> List[NDArray]:
        return particle_temperatures(
            self.excitation_data.energy_per_particle, self.particle_masses
        )

    @property
    def initial_state(self) -> SimulationState:
        electron_hole_density = self.initial_electron_hole_density * np.ones_like(
            self.grid
        )
        temperatures = self.initial_temperatures
        particle_attrs = self.particle_attributes
        return SimulationState.create(
            [
                ParticleState.create(
                    electron_hole_density, temperatures[0], particle_attrs[0]
                ),
                ParticleState.create(
                    electron_hole_density, temperatures[1], particle_attrs[1]
                ),
            ],
            np.zeros_like(self.grid),
        )

    def create_simulation(self) -> Tuple[Simulation, SimulationState]:
        x = self.grid
        N = len(x)

        particle_attrs = self.particle_attributes

        excitation_data = self.excitation_data
        electron_hole_density = self.initial_electron_hole_density * np.ones_like(x)
        final_temperatures = self.excitation_temperatures
        excited_final_state = [  # state immediately after excitation in the absence of transport (i.e. max density)
            ParticleState.create(
                excitation_data.number_density, final_temperatures[0], particle_attrs[0]
            ),
            ParticleState.create(
                excitation_data.number_density, final_temperatures[1], particle_attrs[1]
            ),
        ]

        Ceh = np.array(
            [
                heat_capacity_from_state(attrs, state)
                for (attrs, state) in zip(particle_attrs, excited_final_state)
            ]
        ).sum(axis=0)
        geh = np.average(Ceh) / self.electron_hole_thermalization_time
        gph = np.average(Ceh) / self.carrier_phonon_thermalization_time

        source_fn = create_source_term_T_rise(
            N,
            particle_attrs,
            excitation_data.number_density,
            [final_temperatures[0], final_temperatures[1]],
            self.excitation_time_fwhm,
            self.excitation_time_zero,
            geh,
            gph,
            self.phonon_temperature,
        )
        simul = Simulation(
            N,
            self.resolution,
            SI,
            self.relative_permittivity,
            particle_attrs,
            source_fn,
        )
        initial_electric_field = np.zeros_like(x)
        initial_temperatures = self.initial_temperatures
        initial_state = SimulationState.create(
            [
                ParticleState.create(
                    electron_hole_density, initial_temperatures[0], particle_attrs[0]
                ),
                ParticleState.create(
                    electron_hole_density, initial_temperatures[1], particle_attrs[1]
                ),
            ],
            initial_electric_field,
        )
        return simul, initial_state

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(config: dict) -> SimulationConfig:
        return SimulationConfig(**config)


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


def gam_wavevector_power(a, n, kz, gam):
    def g(x):
        v = np.maximum(1e-3 * np.ones_like(x), 1 + a * kz(x) ** 2)
        return gam * v**n

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
    else:
        raise ValueError("Unknown gam_expr.")


@dataclass(frozen=True)
class OpticalModel:
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

    def create_optical_model_fn(self, simulconf, state_t, states):
        mc, mv = self.mc * SI.m_e, self.mv * SI.m_e
        x = simulconf.grid
        E = electric_field_to_grid(np.array([st.electric_field for st in states]))
        kvectors = integrated_wavevector(E, state_t, self.tau_eph)
        om = 2 * np.pi * SI.c_0 / self.probe_wvl
        alpha = self.alpha_eV / SI.e
        eps_i = partial(create_eps_from, self.eps_v, self.n_val, om)

        def create_model(index: int):
            state = states[index]
            kz = interp1d(x, kvectors[:, index], kind="cubic")
            mx = choose_mass_function(self.expr_mass_x, mc, mv, alpha, kz)
            mz = choose_mass_function(self.expr_mass_z, mc, mv, alpha, kz)
            gx = choose_gam_function(self.expr_gam_x, self.gam_drude, kz)
            gz = choose_gam_function(self.expr_gam_z, self.gam_drude, kz)
            n = interp1d(x, state.number_density[0] * self.density_scale)
            eps_x = eps_i(n, mx, gx)
            eps_z = eps_i(n, mz, gz)
            return (
                eps_x,
                eps_z,
                {"mx": mx, "mz": mz, "gx": gx, "gz": gz, "kz": kz},
            )

        return create_model

    def create_stack_fn(self, simulconf, state_t, states):
        get_model = self.create_optical_model_fn(simulconf, state_t, states)
        L = simulconf.grid[-1] * 0.9999

        def create_stack_from_index(index: int):
            eps_x, eps_z, pars = get_model(index)
            stack = create_stack(L, self.eps_v, eps_x, eps_z)
            return stack, pars

        return create_stack_from_index

    def calculate_reflectivity(self):
        om = 2 * np.pi * SI.c_0 / self.probe_wvl
        simulconf = self.simulation_config
        state_t, states = self.simulation_states
        get_stack = self.create_stack_fn(simulconf, state_t, states)

        def run(index: int):
            stack, pars = get_stack(index)
            Rs = solve_stack(stack, om, self.aoi, "s").R
            Rp = solve_stack(stack, om, self.aoi, "p").R
            ti = state_t[index] - simulconf.excitation_time_zero
            return ti, Rs, Rp, (stack, pars)

        idx = select_nearest_simulation_states(
            simulconf.excitation_time_zero,
            simulconf.excitation_time_fwhm,
            state_t,
        )
        return map(run, idx)


def run_and_append_to_file(model: OpticalModel, save_file) -> list:
    if pathlib.Path(save_file).exists():
        with open(save_file, "r") as io:
            saved_data = json.load(io)
        saved_settings = saved_data.get("settings", [])
        saved_results = saved_data.get("results", [])
        for settings, result in zip(saved_settings, saved_results):
            calculator = OpticalModel(**settings)
            if model == calculator:
                return result
    else:
        saved_settings = []
        saved_results = []
    t, Rs, Rp, _ = unzip(model.calculate_reflectivity())
    result = [list(t), list(Rs), list(Rp)]
    saved_results.append(result)
    saved_settings.append(asdict(model))
    with open(save_file, "w") as io:
        json.dump(
            {
                "settings": saved_settings,
                "results": saved_results,
            },
            io,
        )
    return result

# %%
