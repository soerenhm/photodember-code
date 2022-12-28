from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import cached_property
import gc
import pathlib
import numpy as np
import scipy as sp
from scipy.special import erf

from typing import List, Tuple
from numpy.typing import NDArray

from photodember.src.constants import SI
from photodember.src.transport.fdtable import FermiDiracTable
from photodember.src.transport.fdinterp import interpolate_particle_attributes
from photodember.src.transport.core import ParticleAttributes
from photodember.src.transport.simulation import *

# -----------------------------------------------------------------------------
# Simulation functions

def advance(pde: OdeFunction, init_state: SimulationState, ti: float, dt: float) -> SimulationState:
    gc.collect() # sometimes memory consumption keeps rising, dunno why... this helps...
    return solve_pde(pde, init_state, [ti, ti + dt])[-1]

def save_state_in(f, state: SimulationState, t: float):
    s = np.append([t], state.array).astype(np.float64).tobytes()
    f.write(s)

def simulation_step(f, pde: OdeFunction, init_state: SimulationState, ti: float, tf: float):
    new_state = advance(pde, init_state, ti, tf-ti)
    save_state_in(f, new_state, tf)
    return new_state

def read_simulation_file(filename: str, initial_state: SimulationState) -> Tuple[List[float], List[SimulationState]]:
    t = []
    states = []
    chunksize = len(initial_state.array) + 1
    with open(filename, "rb") as io:
        content = np.frombuffer(io.read(), dtype=np.float64)
        start, stop = 0, chunksize
        while stop <= len(content):
            section = content[start:stop]
            t.append(section[0])
            state = SimulationState(section[1:], initial_state.grid_points, initial_state.number_particles)
            states.append(state)
            start, stop = stop, stop + chunksize
    return t, states

def run_simulation(pde: OdeFunction, save_file: str, init_state: SimulationState, times: ArrayLike):
    times = np.atleast_1d(times)
    t = list(times)
    ti = t.pop(0)
    # Scan file for existing entries
    if pathlib.Path(save_file).exists():
        read_t, read_states = read_simulation_file(save_file, init_state)
        if len(read_t) > 0:
            if ti == read_t[0] and all(np.isclose(init_state.array, read_states[0].array)):
                ti = read_t[-1]
                init_state = read_states[-1]
                t = list(filter(lambda t: t > ti, times))
            else:
                raise ValueError("Simulation file already exists but saved states differ")
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
    energy_per_particle = data[:,2] * SI.e
    return ExcitationData(position, number_density, energy_per_particle)

def to_simulation_grid(data: ExcitationData, grid_coordinates: NDArray) -> ExcitationData:
    number_density = sp.interpolate.interp1d(data.position, data.number_density, kind="cubic")(grid_coordinates)
    energy_per_particle = sp.interpolate.interp1d(data.position, data.energy_per_particle, kind="cubic")(grid_coordinates)
    return ExcitationData(grid_coordinates, number_density, energy_per_particle)

def particle_temperatures(energy_per_particle: NDArray, particle_masses: List[float]) -> List[NDArray]:
    inverse_masses = np.divide(1.0, particle_masses)
    temperature = energy_per_particle * (2/3) / SI.kB
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
        return FermiDiracTable.load_meta(self.source.replace(".csv", ".json"))["relativeMass"] * SI.m_e

def set_mobility(table: FermiDiracTable, meta: dict, new_value: float) -> Tuple[FermiDiracTable, dict]:
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
    return heat_capacity(np.zeros(state.grid_points), attrs.number_density, attrs.energy_density, state.temperature, state.red_chemical_potential)

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

def create_source_term_T_rise(N: int, particles: List[ParticleAttributes], final_density: NDArray, final_temperatures: List[NDArray], tfwhm: float, t0: float, geh: float, gph: float, Tph: float) -> OdeFunction:
    assert len(particles) == 2
    indexer = StateIndexer(N, 2)
    elec_attrs, hole_attrs = particles
    elec = indexer.particle(0)
    hole = indexer.particle(1)
    out = np.zeros(indexer.length)
    G, dTe, dTh, He, Hh = [np.zeros(N) for _ in range(5)]
    
    A = np.sqrt(4*np.log(2)/np.pi)/tfwhm
    def G_(t):
        G[:] = final_density * A * np.exp(-4*np.log(2)*((t-t0)/tfwhm)**2)
        return G

    def dTeh(t):
        dTe[:] = (final_temperatures[0] - Tph) * A * np.exp(-4*np.log(2)*((t-t0)/tfwhm)**2)
        dTh[:] = (final_temperatures[1] - Tph) * A * np.exp(-4*np.log(2)*((t-t0)/tfwhm)**2)
        return dTe, dTh

    def rise(t):
        return .5*(1 + erf(np.sqrt(4*np.log(2))*(t-t0)/tfwhm))

    def S(t, y):
        out.fill(0.0)
        
        G = G_(t)
        dTe, dTh = dTeh(t)
        out[elec.number_density] += G
        out[hole.number_density] += G

        # Electron
        Te, etae = y[elec.temperature], y[elec.red_chemical_potential]
        Ue, Ne = elec_attrs.energy_density, elec_attrs.number_density
        He[:] = (Ue.dT(Te, etae) - Ue.deta(Te, etae)*Ne.dT(Te, etae)/Ne.deta(Te, etae)) * dTe + Ue.deta(Te, etae)/Ne.deta(Te, etae) * G

        # Hole
        Th, etah = y[hole.temperature], y[hole.red_chemical_potential]
        Uh, Nh = hole_attrs.energy_density, hole_attrs.number_density
        Hh[:] = (Uh.dT(Th, etah) - Uh.deta(Th, etah)*Nh.dT(Th, etah)/Nh.deta(Th, etah)) * dTh + Uh.deta(Th, etah)/Nh.deta(Th, etah) * G

        r = rise(t)
        out[elec.energy_density] += He - r*(geh*(Te - Th) + gph*(Te - Tph))
        out[hole.energy_density] += Hh - r*(geh*(Th - Te) + gph*(Th - Tph))
        return out
    return S


@dataclass(frozen=True)
class DielectricConfig:
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
        return [load_particle_attributes(self.electron_data),
                load_particle_attributes(self.hole_data)]
    
    @cached_property
    def excitation_data(self) -> ExcitationData:
        return to_simulation_grid(load_excitation_data(self.excitation_datafile), self.grid)

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
        return particle_temperatures(self.excitation_data.energy_per_particle, self.particle_masses)

    @property
    def initial_state(self) -> SimulationState:
        electron_hole_density = self.initial_electron_hole_density * np.ones_like(self.grid)
        temperatures = self.initial_temperatures
        particle_attrs = self.particle_attributes
        return SimulationState.create(
                   [ParticleState.create(electron_hole_density, temperatures[0], particle_attrs[0]),
                    ParticleState.create(electron_hole_density, temperatures[1], particle_attrs[1])],
                    np.zeros_like(self.grid))

    def create_simulation(self) -> Tuple[Simulation, SimulationState]:
        x = self.grid
        N = len(x)

        particle_attrs = self.particle_attributes

        excitation_data = self.excitation_data
        electron_hole_density = self.initial_electron_hole_density * np.ones_like(x)
        final_temperatures = self.excitation_temperatures
        excited_final_state = [ # state immediately after excitation in the absence of transport (i.e. max density)
            ParticleState.create(excitation_data.number_density, final_temperatures[0], particle_attrs[0]),
            ParticleState.create(excitation_data.number_density, final_temperatures[1], particle_attrs[1])
        ]

        Ceh = np.array([heat_capacity_from_state(attrs, state) for (attrs, state) in zip(particle_attrs, excited_final_state)]).sum(axis=0)
        geh = np.average(Ceh) / self.electron_hole_thermalization_time
        gph = np.average(Ceh) / self.carrier_phonon_thermalization_time
        
        source_fn = create_source_term_T_rise(N, particle_attrs, excitation_data.number_density, [final_temperatures[0], final_temperatures[1]], self.excitation_time_fwhm, self.excitation_time_zero, geh, gph, self.phonon_temperature)
        simul = Simulation(N, self.resolution, SI, self.relative_permittivity, particle_attrs, source_fn)
        initial_electric_field = np.zeros_like(x)
        initial_temperatures = self.initial_temperatures
        initial_state = SimulationState.create(
                   [ParticleState.create(electron_hole_density, initial_temperatures[0], particle_attrs[0]),
                    ParticleState.create(electron_hole_density, initial_temperatures[1], particle_attrs[1])],
                    initial_electric_field)
        return simul, initial_state

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(config: dict) -> DielectricConfig:
        return DielectricConfig(**config)