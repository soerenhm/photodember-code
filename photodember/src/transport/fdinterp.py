from __future__ import annotations
from dataclasses import dataclass
from functools import partial, wraps
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline

from .fdtable import DataSchema, FermiDiracTable
from .core import ParticleAttributes, StateFunction


Interpolation = Callable[[FermiDiracTable, str], StateFunction]


def cache_carrier_state_func(func: StateFunction):
    """Caches the result of an expensive CarrierStateFunc call."""
    cache = [np.array([]) for _ in range(3)]

    @wraps(func)
    def inner(T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        nonlocal cache
        T = np.atleast_1d(T)
        eta = np.atleast_1d(eta)
        if T.shape == eta.shape == cache[0].shape == cache[1].shape == cache[2].shape:
            I = np.logical_or(T != cache[0], eta != cache[1])
            cache[0][I] = np.copy(T[I])
            cache[1][I] = np.copy(eta[I])
            cache[2][I] = func(T[I], eta[I])
        else:
            cache[0] = np.copy(T)
            cache[1] = np.copy(eta)
            cache[2] = func(T, eta)  # type: ignore
        return cache[2]

    return inner


@dataclass(frozen=True)
class CubicLogInterp:
    """Bicubic interpolation of log of kinetic integrals 0, 1, and 2."""

    _spline: RectBivariateSpline

    @staticmethod
    def interpolate(table: FermiDiracTable, property: str) -> CubicLogInterp:
        T = table.temperature
        eta = table.reduced_chemical_potential
        qty = np.log(table.as_array2d(property))
        return CubicLogInterp(RectBivariateSpline(T, eta, qty, kx=3, ky=3, s=0))

    def eval(self, T: ArrayLike, eta: ArrayLike, dT=0, deta=0, grid=False) -> ArrayLike:
        log_value = self._spline(T, eta, dx=0, dy=0, grid=grid)
        if dT == 0 and deta == 0:
            return np.exp(log_value)
        elif (dT == 1 and deta == 0) or (dT == 0 and deta == 1):
            return np.exp(log_value) * self._spline(T, eta, dx=dT, dy=deta, grid=grid)
        else:
            raise NotImplementedError(
                "Derivatives of order higher than 1 are not yet implemented!"
            )

    def __call__(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return self.eval(T, eta)

    def dT(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return self.eval(T, eta, dT=1)

    def deta(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return self.eval(T, eta, deta=1)


@dataclass
class CubicLogInterpFastDeriv:
    """Bicubic interpolation of log of kinetic integrals 0, 1, and 2 but with faster derivatives."""

    _spl: RectBivariateSpline
    _spl_dT: RectBivariateSpline
    _spl_deta: RectBivariateSpline

    def __post_init__(self):
        self.ev = cache_carrier_state_func(self.ev)  # type: ignore
        self.dT = cache_carrier_state_func(self.dT)  # type: ignore
        self.deta = cache_carrier_state_func(self.deta)  # type: ignore

    @staticmethod
    def interpolate(table: FermiDiracTable, property: str) -> CubicLogInterpFastDeriv:
        T = table.temperature
        eta = table.reduced_chemical_potential
        grid = np.meshgrid(T, eta, indexing="ij")
        qty = np.log(table.as_array2d(property))
        spl = RectBivariateSpline(T, eta, qty, kx=3, ky=3, s=0)
        spl_dT = RectBivariateSpline(
            T, eta, spl(*grid, dx=1, grid=False), kx=3, ky=3, s=0
        )
        spl_deta = RectBivariateSpline(
            T, eta, spl(*grid, dy=1, grid=False), kx=3, ky=3, s=0
        )
        return CubicLogInterpFastDeriv(spl, spl_dT, spl_deta)

    def __call__(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return self.ev(T, eta)

    def ev(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return np.exp(self._spl(T, eta, grid=False))

    def dT(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return self.ev(T, eta) * self._spl_dT(T, eta, grid=False)  # type: ignore

    def deta(self, T: ArrayLike, eta: ArrayLike) -> ArrayLike:
        return self.ev(T, eta) * self._spl_deta(T, eta, grid=False)  # type: ignore


DEFAULT_FD_INTERPOLATOR = CubicLogInterpFastDeriv.interpolate


def interpolate_particle_attributes(
    table: FermiDiracTable,
    electric_charge: float,
    kB: float,
    method: Interpolation = DEFAULT_FD_INTERPOLATOR,
):
    interpolate = partial(method, table)
    N = interpolate(DataSchema.NUMBER_DENSITY)
    U = interpolate(DataSchema.ENERGY_DENSITY)
    I0 = interpolate(DataSchema.KINETIC_INTEGRAL_0)
    I1 = interpolate(DataSchema.KINETIC_INTEGRAL_1)
    I2 = interpolate(DataSchema.KINETIC_INTEGRAL_2)
    q = electric_charge
    L11 = lambda T, eta: I2(T, eta) - (kB * T * eta) * I1(T, eta)
    L12 = lambda T, eta: q * I1(T, eta)  # type: ignore
    L21 = lambda T, eta: q * (I1(T, eta) - kB * T * eta * I0(T, eta))
    L22 = lambda T, eta: q**2 * I0(T, eta)  # type: ignore
    return ParticleAttributes(electric_charge, N, U, L11, L12, L21, L22)
