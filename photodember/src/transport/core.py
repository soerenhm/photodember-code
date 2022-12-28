"""
One dimensional particle and energy transport.

Densities and currents are evaluated on a staggered grid.
- Number and kinetic-energy densities are evaluated on integer grid points,
i.e. i = 0, 1, 2, ...
- Current densities are evaluated on half-integer points, i.e.
i = -1/2, 1/2, 3/2, ...

Graphically, the indexing scheme looks like this:
    * | * | * | ... * | * |
where vertical bars (|) represent integers and the stars (*) represent half integers.

The boundary conditions conserve densities, which is enforced by setting the 
current at the left/right boundaries to zero. (The current at the right 
boundary lies outside the grid and is ignored since it's anyway zero.)

Most of functions for calculating densities/currents/etc. compute the values 
in place, i.e. the values are stored in the array passed as the first 
argument. This avoids unnecessary allocations which can be expensive when many 
time steps are required for solving PDE. Just be aware that these functions 
generally mutate the first argument.
"""

from dataclasses import dataclass
import numba
import numpy as np
import scipy.optimize as optim
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Protocol, Tuple


# -----------------------------------------------------------------------------
# Types

ArrayF64 = NDArray[np.float64]

KineticCoefficient = Callable[[ArrayLike, ArrayLike], ArrayLike]


class StateFunction(Protocol):
    def __call__(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        ...

    def deta(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        ...

    def dT(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        ...


@dataclass(frozen=True)
class ParticleAttributes:
    electric_charge: float
    number_density: StateFunction
    energy_density: StateFunction
    L11: KineticCoefficient
    L12: KineticCoefficient
    L21: KineticCoefficient
    L22: KineticCoefficient


# -----------------------------------------------------------------------------
# Current densities


@numba.njit
def energy_current_density(
    u: ArrayF64,
    dx: float,
    e: float,
    T: ArrayF64,
    eF: ArrayF64,
    E: ArrayF64,
    L11: ArrayF64,
    L12: ArrayF64,
) -> ArrayF64:
    """In-place calculation of the kinetic-energy current density."""
    u[0] = 0.0
    for i in range(1, len(u)):
        T_dx = (T[i] - T[i - 1]) / dx
        eF_dx = (eF[i] - eF[i - 1]) / dx
        L12i = 0.5 * (L12[i] + L12[i - 1])
        u[i] = -0.5 * (L11[i] / T[i] + L11[i - 1] / T[i - 1]) * T_dx + (
            L12i * (eF_dx / e + E[i])
        )
    return u


@numba.njit
def charge_current_density(
    J: ArrayF64,
    dx: float,
    e: float,
    T: ArrayF64,
    eF: ArrayF64,
    E: ArrayF64,
    L21: ArrayF64,
    L22: ArrayF64,
) -> ArrayF64:
    """In-place calculation of the charge current density."""
    J[0] = 0.0
    for i in range(1, len(J)):
        T_dx = (T[i] - T[i - 1]) / dx
        eF_dx = (eF[i] - eF[i - 1]) / dx
        L22i = 0.5 * (L22[i] + L22[i - 1])
        J[i] = -0.5 * (L21[i] / T[i] + L21[i - 1] / T[i - 1]) * T_dx + (
            L22i * (eF_dx / e + E[i])
        )
    return J


# -----------------------------------------------------------------------------
# Differentials


@numba.njit
def number_density_differntial(
    dN: ArrayF64, dt: float, dx: float, q: float, J: ArrayF64
) -> ArrayF64:
    """In-place calculation of the number-density differential."""
    dt_by_qdx = dt / dx / q
    for i in range(1, len(J) - 1):
        dN[i] = (J[i] - J[i + 1]) * dt_by_qdx
    dN[0] = -J[1] * dt_by_qdx
    dN[-1] = J[-1] * dt_by_qdx
    return dN


@numba.njit
def energy_density_differential(
    dU: ArrayF64, dt: float, dx: float, u: ArrayF64, J: ArrayF64, E: ArrayF64
) -> ArrayF64:
    """In-place calculation of the energy-density differential."""
    dt_by_dx = dt / dx
    for i in range(1, len(u) - 1):
        dQ = 0.5 * (E[i] * J[i] + E[i + 1] * J[i + 1]) * dt
        dU[i] = (u[i] - u[i + 1]) * dt_by_dx + dQ
    dU[0] = -u[1] * dt_by_dx + 0.5 * E[1] * J[1] * dt
    dU[-1] = u[-1] * dt_by_dx + 0.5 * E[-1] * J[-1] * dt
    return dU


def heat_capacity(
    C: ArrayF64, N: StateFunction, U: StateFunction, T: ArrayF64, eta: ArrayF64
) -> ArrayF64:
    C[:] = U.dT(T, eta) - U.deta(T, eta) * N.dT(T, eta) / N.deta(T, eta)  # type: ignore
    return C


# -----------------------------------------------------------------------------
# Change of variables


def jacobian_from_densities(
    jac: ArrayF64, N: StateFunction, U: StateFunction, T: ArrayLike, eta: ArrayLike
) -> ArrayF64:
    """Calculates the Jacobian, transforming densities (N, U) to variables (T, eta)."""
    jac[:, 0, 0] = N.dT(T, eta)
    jac[:, 0, 1] = N.deta(T, eta)
    jac[:, 1, 0] = U.dT(T, eta)
    jac[:, 1, 1] = U.deta(T, eta)
    return jac


@numba.njit
def solve_2by2_inplace(out: ArrayF64, matrix: ArrayF64, x: ArrayF64):
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    det = a * d - b * c
    if det == 0:
        print(f"Warning: Jacobian is not invertible")
        return out
    else:
        idet = 1.0 / det
        out[0] = idet * (d * x[0] - b * x[1])
        out[1] = idet * (-c * x[0] + a * x[1])
        return out


@numba.njit
def transform_density_differentials(
    dT: ArrayF64, deta: ArrayF64, jac: ArrayF64, dN: ArrayF64, dU: ArrayF64
) -> Tuple[ArrayF64, ArrayF64]:
    """Calculates the differentials of (dT, deta) given known differentials (dN, dU) and the Jacobian transformation (N, U) -> (T, eta)."""
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    for i in range(jac.shape[0]):
        jaci = jac[i, :, :]
        y[0] = dN[i]
        y[1] = dU[i]
        solve_2by2_inplace(x, jaci, y)
        dT[i] = x[0]
        deta[i] = x[1]
    return dT, deta


def fermi_energy(
    eF: ArrayF64, eta: ArrayF64, T: ArrayF64, kB: float, electric_charge: float
) -> ArrayF64:
    s = -np.sign(electric_charge)
    eF[:] = eta * (s * kB * T)
    return eF


# -----------------------------------------------------------------------------
# Electric field


@numba.njit
def electric_field(
    E: ArrayF64, dx: float, eps_0: float, eps_rel: float, rho: ArrayF64
) -> ArrayF64:
    """Electric field from Gauss' law."""
    k = dx / (eps_0 * eps_rel)
    acc = 0.0
    E[0] = 0.0
    for i in range(1, len(E)):
        ei = k * rho[i - 1]
        E[i] = ei + acc
        acc += ei
    return E


@numba.njit
def electric_potential(
    phi: ArrayF64, dx: float, eps_0: float, eps_rel: float, rho: ArrayF64
) -> ArrayF64:
    k = -2.0 * dx**2 / (eps_0 * eps_rel)
    phi[-1] = phi[-2] = 0.0
    for i in range(len(phi) - 2, -1, -1):
        phi[i] = k * rho[i + 1] - phi[i + 2] + 2 * phi[i + 1]
    return phi


@numba.njit
def ambipolar_electric_field(
    E, dx: float, q: float, L21e, L21h, L22e, L22h, Te, Th, zeta_e, zeta_h
):
    E[0] = 0.0
    for i in range(1, len(E) - 1):
        dTe = (Te[i] - Te[i - 1]) / dx
        dTh = (Th[i] - Th[i - 1]) / dx
        dzeta_e = (zeta_e[i] - zeta_e[i - 1]) / dx
        dzeta_h = (zeta_h[i] - zeta_h[i - 1]) / dx
        L22ei = 0.5 * (L22e[i] + L22e[i - 1])
        L22hi = 0.5 * (L22h[i] + L22h[i - 1])
        E[i] = (
            0.5 * (L21e[i] / Te[i] + L21e[i - 1] / Te[i - 1]) * dTe
            + 0.5 * (L21h[i] / Th[i] + L21h[i - 1] / Th[i - 1]) * dTh
        ) / (L22ei + L22hi) - (L22ei * dzeta_e + L22hi * dzeta_h) / q / (L22ei + L22hi)
    return E


# -----------------------------------------------------------------------------
# Solve T, eta


def solve_reduced_chemical_potential(
    number_density_func: StateFunction,
    T: float,
    number_density: float,
    bounds: Tuple[float, float],
) -> float:
    f = lambda x: (np.log(number_density) - np.log(number_density_func(T, x))) ** 2
    result = optim.minimize_scalar(f, bounds, method="golden", tol=1e-14)
    if not result.success:
        print(
            f"Minimization did not reach the desired tolerance; `eta` for `T = {T}` and `N = {number_density}` may be unreliable."
        )
    return np.atleast_1d(result.x)[0]


def solve_temperature(
    energy_density_func: StateFunction,
    eta: float,
    energy_density: float,
    bounds: Tuple[float, float],
) -> float:
    f = lambda x: (np.log(energy_density) - np.log(energy_density_func(x, eta))) ** 2
    result = optim.minimize_scalar(f, bounds, method="golden", tol=1e-14)
    if not result.success:
        print(
            f"Minimization did not reach the desired tolerance; `T` for `eta = {eta}` and `U = {energy_density}` may be unreliable."
        )
    return np.atleast_1d(result.x)[0]
