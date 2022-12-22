from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Optional, Tuple

from ..constants import PhysicalConstants
from .core import *

# -----------------------------------------------------------------------------
# Types

OdeFunction = Callable[[float, ArrayF64], ArrayF64]

# -----------------------------------------------------------------------------
# Indexing

@dataclass(frozen=True)
class PropertyIndexer:
    """
    Indexing scheme for properties of a charged particle.

    On a spatial grid with `N` points, the state of a particle is stored as 
    a `4N` array, which consistes of four `N` chunks, which represents
    (in order) the particles'...
    - Number density
    - Internal energy density
    - Reduced chemical potential
    - Temperature
    """
    grid_points: int
    offset: int

    @staticmethod
    def eachparticle(grid_points: int, number_particles: int):
        for n in range(number_particles):
            offset = 4 * grid_points * n
            yield PropertyIndexer(grid_points, offset)

    def unpack(self) -> Tuple[int, int]:
        return self.grid_points, self.offset

    @property
    def number_density(self) -> slice:
        return slice(self.offset, self.offset + self.grid_points)

    @property
    def energy_density(self) -> slice:
        n, o = self.unpack()
        return slice(o + n, o + 2*n)

    @property
    def red_chemical_potential(self) -> slice:
        n, o = self.unpack()
        return slice(o + 2*n, o + 3*n)

    @property
    def temperature(self) -> slice:
        n, o = self.unpack()
        return slice(o + 3*n, o + 4*n)

    @property
    def view(self) -> slice:
        n, o = self.unpack()
        return slice(o, o + 4*n)

    @property
    def L11(self) -> slice:
        n = self.grid_points
        return slice(0, n)
    
    @property
    def L12(self) -> slice:
        n = self.grid_points
        return slice(n, 2*n)

    @property
    def L21(self) -> slice:
        n = self.grid_points
        return slice(2*n, 3*n)

    @property
    def L22(self) -> slice:
        n = self.grid_points
        return slice(3*n, 4*n)


@dataclass(frozen=True)
class StateIndexer:
    """
    Indexing scheme for the state of a simulation, i.e. the state of all 
    particles and the internal electric field.

    On a spatial grid with `N` points and `M` particles, the state is stored 
    as an `N (4M + 1)` array. The first `M` chunks of length `4N` each 
    represents the state of each particle (see `PropertyIndexer` above). 
    The final `N` chunk holds the electric field.
    """
    grid_points: int
    number_particles: int

    def particle(self, index: int) -> PropertyIndexer:
        n = self.grid_points
        o = n * 4 * index
        return PropertyIndexer(n, o)

    def unpack(self) -> Tuple[int, int]:
        return self.grid_points, self.number_particles

    @property
    def length(self) -> int:
        return self.grid_points * self.number_particles * 4 + self.grid_points

    @property
    def particles(self):
        return PropertyIndexer.eachparticle(self.grid_points, self.number_particles)

    @property
    def electric_field(self) -> slice:
        o = self.grid_points * 4 * self.number_particles
        return slice(o, o + self.grid_points)

# -----------------------------------------------------------------------------
# Simulation states

@dataclass
class ParticleState:
    """
    Holds the state of a particle in a one-dimensional array. 
    
    See also `PropertyIndexer`.
    """
    array: ArrayF64
    grid_points: int

    @staticmethod
    def create(number_density: ArrayF64, temperature: ArrayF64, particle_attrs: ParticleAttributes) -> ParticleState:
        if len(number_density) != len(temperature):
            raise ValueError(f"`number_density` and `energy_density` must be arrays of the same length!")
        npts = len(number_density)
        array = np.zeros(4*npts)
        indexer = PropertyIndexer(npts, 0)
        array[indexer.number_density] = np.copy(number_density)
        array[indexer.temperature] = np.copy(temperature)
        eta = np.array([solve_reduced_chemical_potential(particle_attrs.number_density, Ti, Ni, bounds=(-200., 200.)) for (Ti, Ni) in zip(temperature, number_density)])
        array[indexer.red_chemical_potential] = eta
        array[indexer.energy_density] = particle_attrs.energy_density(temperature, eta)
        return ParticleState(array=array, grid_points=npts)

    @property
    def indexer(self):
        return PropertyIndexer(self.grid_points, 0)

    @property
    def number_density(self) -> ArrayF64:
        return self.array[self.indexer.number_density]

    @property
    def energy_density(self) -> ArrayF64:
        return self.array[self.indexer.energy_density]

    @property
    def red_chemical_potential(self) -> ArrayF64:
        return self.array[self.indexer.red_chemical_potential]

    @property
    def temperature(self) -> ArrayF64:
        return self.array[self.indexer.temperature]
    
    def chemical_potential(self, q: float, kB: float):
        return fermi_energy(np.zeros(self.grid_points), self.red_chemical_potential, self.temperature, kB, q)


@dataclass
class SimulationState:
    """
    Holds the state of the simulation, i.e. all particles and the internal 
    electric field in a one-dimensional array. 
    
    See also `StateIndexer`.
    """
    array: ArrayF64
    grid_points: int
    number_particles: int

    @staticmethod
    def create(states: List[ParticleState], electric_field: ArrayF64) -> SimulationState:
        npts = states[0].grid_points
        error = ValueError("Array lengths are incompatiable!")
        if npts != len(electric_field): raise error
        def join_states(acc: ArrayF64, state: ParticleState) -> ArrayF64:
            if len(acc) == 0:
                return np.copy(state.array)
            elif len(state.array) == 4 * npts:
                return np.append(acc, np.copy(state.array))
            else:
                raise error
        array = reduce(join_states, states, np.array([]))
        array = np.append(array, np.copy(electric_field))
        return SimulationState(array, npts, len(states))

    @property
    def indexer(self) -> StateIndexer:
        return StateIndexer(self.grid_points, self.number_particles)

    def particle(self, index: int):
        I = self.indexer.particle(index).view
        return ParticleState(self.array[I], self.grid_points)

    @property
    def particles(self):
        for n in range(self.number_particles):
            yield self.particle(n)

    @property
    def number_density(self):
        return [p.number_density for p in self.particles]

    @property
    def energy_density(self):
        return [p.energy_density for p in self.particles]

    @property
    def temperature(self):
        return [p.temperature for p in self.particles]

    @property
    def red_chemical_potential(self):
        return [p.red_chemical_potential for p in self.particles]

    def chemical_potential(self, q: float, kB: float):
        return [p.chemical_potential(q, kB) for p in self.particles]

    @property
    def electric_field(self) -> ArrayF64:
        return self.array[self.indexer.electric_field]

# -----------------------------------------------------------------------------
# Simulation class

@dataclass
class Simulation:
    grid_points: int
    resolution: float
    const: PhysicalConstants
    eps_rel: float
    particles: list[ParticleAttributes]
    source_fn: Optional[OdeFunction] = None

    def create_pde(self):
        particles = self.particles
        e, kB, eps_0 = self.const.e, self.const.kB, self.const.eps_0
        npts, dx = self.grid_points, self.resolution

        state_indexer = StateIndexer(npts, len(particles))
        property_indexer = PropertyIndexer(npts, 0)

        # Pre-allocate arrays
        dy = np.zeros(state_indexer.length)
        S = np.zeros_like(dy)
        Lij = np.zeros_like(dy)
        eF, J, u, dN, dU, dT, deta = [np.zeros(npts) for _ in range(7)]
        jac = np.zeros((npts, 2, 2))

        # Source function
        if self.source_fn is None: 
            source_fn = lambda t, y: S
        else:
            source_fn = self.source_fn

        def compute_dT_deta(particle: ParticleAttributes, T, eta, dN, dU):
            nonlocal jac
            jac = jacobian_from_densities(jac, particle.number_density, particle.energy_density, T, eta)
            return transform_density_differentials(dT, deta, jac, dN, dU)

        def dydt(t, y_: ArrayF64):
            nonlocal dN, dU, J, u, eF, Lij  # nonlocal for arrays may not be needed?
            eta = property_indexer.red_chemical_potential
            T = property_indexer.temperature
            N = property_indexer.number_density
            U = property_indexer.energy_density
            E = state_indexer.electric_field
            L11 = property_indexer.L11
            L12 = property_indexer.L12
            L21 = property_indexer.L21
            L22 = property_indexer.L22
            dy.fill(0.0)

            efield = y_[E]
            S = source_fn(t, y_)

            # Replace with property indexer?
            for particle, inds in zip(particles, state_indexer.particles):
                view = inds.view
                y = y_[view]
                L = Lij[view]
                # Calculate new kinetic coefficients
                L[L11] = particle.L11(y[T], y[eta])
                L[L12] = particle.L12(y[T], y[eta])
                L[L21] = particle.L21(y[T], y[eta])
                L[L22] = particle.L22(y[T], y[eta])

                y = y_[view]
                eF = fermi_energy(eF, y[eta], y[T], kB, particle.electric_charge)

                # Currents
                u = energy_current_density(u, dx, e, y[T], eF, efield, Lij[view][L11], Lij[view][L12])
                J = charge_current_density(J, dx, e, y[T], eF, efield, Lij[view][L21], Lij[view][L22])

                # Differentials
                dN = number_density_differntial(dN, 1.0, dx, particle.electric_charge, J)
                dU = energy_density_differential(dU, 1.0, dx, u, J, efield)
                dN[:] += S[view][N]
                dU[:] += S[view][U]
                dT, deta = compute_dT_deta(particle, y[T], y[eta], dN, dU)
                dT[:] += S[view][T]
                deta[:] += S[view][eta]

                # update dydt
                dy_ = dy[view]
                dy_[N] = dN
                dy_[U] = dU
                dy_[T] = dT
                dy_[eta] = deta
                dy[E] += -J/eps_0/self.eps_rel
            dy[E] += S[E]
            return dy
        return dydt


def solve_pde(pde: OdeFunction, initial_state: SimulationState, times: ArrayLike, **solver_args) -> List[SimulationState]:
    t = np.atleast_1d(times)
    if len(t) < 2: raise ValueError(f"`times` must be a sequence of times with at least an initial and a final value.")
    _solver_args = {"method": "LSODA", "t_eval": t}
    _solver_args.update(solver_args)
    y0 = np.copy(initial_state.array)
    soln = solve_ivp(pde, [t[0], t[-1]], y0, **_solver_args)
    states =  [SimulationState(soln.y[:,i], initial_state.grid_points, initial_state.number_particles) for i, _ in enumerate(t)]
    return states