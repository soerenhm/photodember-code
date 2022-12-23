# %%

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple
import pathlib
import json
import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.special import erf

from photodember.src.constants import SI
from photodember.src.transport.fdtable import FermiDiracTable
from photodember.src.transport.fdinterp import interpolate_particle_attributes
from photodember.src.transport.core import ParticleAttributes
from photodember.src.transport.simulation import *


# -----------------------------------------------------------------------------
# Load excitation data 

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
# Load particle data

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

def create_source_term(N: int, particles: List[ParticleAttributes], final_density: NDArray, tfwhm: float, t0: float, geh: float, gph: float, Tph: float) -> OdeFunction:
    assert len(particles) == 2
    indexer = StateIndexer(N, 2)
    elec_attrs, hole_attrs = particles
    elec = indexer.particle(0)
    hole = indexer.particle(1)
    out = np.zeros(indexer.length)

    A = np.sqrt(4*np.log(2)/np.pi)/tfwhm
    G = np.zeros(indexer.grid_points)
    def d_dt(t):
        G[:] = final_density * A * np.exp(-4*np.log(2)*((t-t0)/tfwhm)**2)
        return G

    def rise(t):
        return .5*(1 + erf(np.sqrt(4*np.log(2))*(t-t0)/tfwhm))

    def S(t, y):
        out.fill(0.0)
        G = d_dt(t)
        out[elec.number_density] += G
        out[hole.number_density] += G

        Te, etae = y[elec.temperature], y[elec.red_chemical_potential]
        Th, etah = y[hole.temperature], y[hole.red_chemical_potential]

        dUe_dNe = elec_attrs.energy_density.deta(Te, etae) / elec_attrs.number_density.deta(Te, etae) # type: ignore
        dUh_dNh = hole_attrs.energy_density.deta(Th, etah) / hole_attrs.number_density.deta(Th, etah) # type: ignore
        
        # if t < t0 + 1.5*tfwhm:
        #     out[elec.energy_density] += G*dUe_dNe #- geh*(Te - Th) - gph*(Te - Tph)
        #     out[hole.energy_density] += G*dUh_dNh #- geh*(Th - Te) - gph*(Th - Tph)
        # else:
        r = rise(t)
        out[elec.energy_density] += G*dUe_dNe - r*(geh*(Te - Th) + gph*(Te - Tph))
        out[hole.energy_density] += G*dUh_dNh - r*(geh*(Th - Te) + gph*(Th - Tph))
        return out
    return S


@dataclass
class FusedSilicaConfig:
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

    @property
    def grid(self) -> NDArray:
        return np.arange(0, self.domain_size, step=self.resolution)

    @property
    def electron_data(self) -> ParticleData:
        return ParticleData(-SI.e, self.electron_mobility, self.fdtable_conduction_band)

    @property
    def hole_data(self) -> ParticleData:
        return ParticleData(SI.e, self.hole_mobility, self.fdtable_valence_band)

    @property
    def particle_attributes(self):
        return [load_particle_attributes(self.electron_data),
                load_particle_attributes(self.hole_data)]

    def create_simulation(self) -> Tuple[Simulation, SimulationState]:
        x = self.grid
        N = len(x)

        particle_attrs = self.particle_attributes

        excitation_data = to_simulation_grid(load_excitation_data(self.excitation_datafile), x)
        electron_hole_density = self.initial_electron_hole_density * np.ones_like(x)
        masses = [self.electron_data.mass, self.hole_data.mass]
        temperatures = particle_temperatures(excitation_data.energy_per_particle, masses)
        excited_final_state = [ # state immediately after excitation in the absence of transport (i.e. max density)
            ParticleState.create(excitation_data.number_density, temperatures[0], particle_attrs[0]),
            ParticleState.create(excitation_data.number_density, temperatures[1], particle_attrs[1])
        ]

        Ceh = np.array([heat_capacity_from_state(attrs, state) for (attrs, state) in zip(particle_attrs, excited_final_state)]).sum(axis=0)
        geh = np.average(Ceh) / self.electron_hole_thermalization_time
        gph = np.average(Ceh) / self.carrier_phonon_thermalization_time
        
        source_fn = create_source_term(N, particle_attrs, excitation_data.number_density, self.excitation_time_fwhm, self.excitation_time_zero, geh, gph, self.phonon_temperature)
        simul = Simulation(N, self.resolution, SI, self.relative_permittivity, particle_attrs, source_fn)
        initial_electric_field = np.zeros_like(x)
        initial_state = SimulationState.create(
                   [ParticleState.create(electron_hole_density, temperatures[0], particle_attrs[0]),
                    ParticleState.create(electron_hole_density, temperatures[1], particle_attrs[1])],
                    initial_electric_field)
        return simul, initial_state

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(config: dict) -> FusedSilicaConfig:
        return FusedSilicaConfig(**config)


# %% 

import gc

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
# Simulator

conf = FusedSilicaConfig(
    domain_size = 1e-6,
    resolution = 2e-9,
    initial_electron_hole_density = 1e18,
    excitation_datafile = "photodember/data/Excitation sapphire 2.3 Jcm2.txt",
    excitation_time_fwhm = 50e-15,
    excitation_time_zero = 150e-15,
    fdtable_conduction_band = "photodember/data/SiO2_CB.csv",
    fdtable_valence_band = "photodember/data/SiO2_VB.csv",
    carrier_phonon_thermalization_time = 1e-11,
    electron_hole_thermalization_time = 1e-14,
    electron_mobility = 3.5e-4,
    hole_mobility = 5.8e-5,
    phonon_temperature = 300.0,
    relative_permittivity = 3.8
)
simul, initial_state = conf.create_simulation()
pde = simul.create_pde()

save_file = "photodember/simulations/fs1.dat"
meta_file = save_file.replace(".dat", ".json")
if not pathlib.Path(meta_file).exists():
    with open(save_file.replace(".dat", ".json"), "w") as io:
        json.dump(conf.to_dict(), io)
times = np.linspace(0.0, 1e-12, 1000)
result = run_simulation(pde, save_file, initial_state, times)