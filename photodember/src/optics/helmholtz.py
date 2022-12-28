from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
import scipy.integrate
from numpy.typing import NDArray
from typing import Callable, Union, List, Optional

from photodember.src.constants import SI


# -----------------------------------------------------------------------------
# Types

Number = Union[float, complex]
ArrayF64 = NDArray[np.float64]
PermittivityFunc = Callable[[float], complex]
OdeFunction = Callable[[float, List[Number]], List[Number]]


class Polarization(Enum):
    s = "s"
    p = "p"


class InvalidPolarizationError(Exception):
    def __init__(self, arg):
        message = (
            f"Invalid polarization! Must be of type `Polarization`. Here: {type(arg)}."
        )
        super().__init__(message)


@dataclass(frozen=True)
class IsotropicPermittivity:
    eps: PermittivityFunc
    deriv: PermittivityFunc

    @property
    def eps_o(self) -> PermittivityFunc:
        return self.eps

    @property
    def eps_e(self) -> PermittivityFunc:
        return self.eps

    @property
    def deriv_o(self) -> PermittivityFunc:
        return self.deriv

    @property
    def deriv_e(self) -> PermittivityFunc:
        return self.deriv


@dataclass(frozen=True)
class UniaxialPermittivity:
    eps_o: PermittivityFunc
    eps_e: PermittivityFunc
    deriv_o: PermittivityFunc
    deriv_e: PermittivityFunc


Permittivity = Union[IsotropicPermittivity, UniaxialPermittivity]


# -----------------------------------------------------------------------------
# Helpers

c_0 = SI.c_0


def approx_deriv(f: Callable[[float], Number], x: float, dx: float) -> Number:
    return (f(x + dx) - f(x)) / dx


def is_finite(x: float) -> bool:
    return abs(x) < math.inf


def positive_imag(z: complex) -> complex:
    return z if z.imag >= 0.0 else -z


# -----------------------------------------------------------------------------
# API


@dataclass(frozen=True)
class Layer:
    thickness: float
    perm: Permittivity

    @staticmethod
    def create(
        thickness: float,
        eps_o: PermittivityFunc,
        eps_e: Optional[PermittivityFunc] = None,
        deriv_o: Optional[PermittivityFunc] = None,
        deriv_e: Optional[PermittivityFunc] = None,
    ) -> Layer:
        _deriv_o = (
            (lambda x: approx_deriv(eps_o, x, thickness * 1e-6))
            if deriv_o is None
            else deriv_o
        )
        if eps_e is None:
            return Layer(thickness, IsotropicPermittivity(eps_o, _deriv_o))
        else:
            _deriv_e = (
                (lambda x: approx_deriv(eps_e, x, thickness * 1e-6))
                if deriv_e is None
                else deriv_e
            )
            return Layer(
                thickness,
                UniaxialPermittivity(
                    eps_o=eps_o, eps_e=eps_e, deriv_o=_deriv_o, deriv_e=_deriv_e
                ),
            )

    @property
    def eps_o(self) -> PermittivityFunc:
        return self.perm.eps_o

    @property
    def eps_e(self) -> PermittivityFunc:
        return self.perm.eps_e

    @property
    def deriv_o(self) -> PermittivityFunc:
        return self.perm.deriv_o

    @property
    def deriv_e(self) -> PermittivityFunc:
        return self.perm.deriv_e


@dataclass(frozen=True)
class Stack:
    layers: list[Layer]

    def __post_init__(self):
        layers = self.layers
        if len(layers) < 1:
            raise ValueError("Stack cannot be empty")
        if is_finite(layers[-1].thickness):
            self.layers.append(
                Layer.create(math.inf, lambda _: 1.0 + 0j, deriv_o=lambda _: 0.0 + 0j)
            )
        else:
            x = sum(layer.thickness for layer in self.layers[:-1])
            eps_o = self.layers[-1].eps_o(x)
            eps_e = self.layers[-1].eps_e(x)
            if max(eps_o.imag, eps_e.imag).imag > 1e-12:
                raise ValueError(
                    f"Imaginary part of epsilon at the rightmost boundary (at x = {x}) is {(eps_o.imag, eps_e.imag)}; it must be zero."
                )

    @property
    def length(self) -> float:
        layers = self.layers[:-1]
        return 0.0 if len(layers) == 0 else sum(layer.thickness for layer in layers)

    @property
    def first(self) -> Layer:
        return self.layers[0]

    @property
    def last(self) -> Layer:
        return self.layers[-1]


# -----------------------------------------------------------------------------
# Core


@dataclass(frozen=True)
class KVector:
    k0: float
    kx: float

    @staticmethod
    def create(omega: float, theta: float) -> "KVector":
        k0 = omega / c_0
        kx = k0 * np.sin(np.deg2rad(theta))
        return KVector(k0=k0, kx=kx)

    def kz(self, eps_o: complex, eps_e: complex, pol: Polarization) -> complex:
        if pol is Polarization("p"):
            return positive_imag(
                np.sqrt(eps_o * self.k0**2.0 - self.kx**2.0 * (eps_o / eps_e) + 0j)
            )
        else:
            return positive_imag(np.sqrt(eps_o * self.k0**2.0 - self.kx**2.0 + 0j))


def de_dz(
    de: list[complex],
    z: float,
    e: list[complex],
    perm: Permittivity,
    k0: float,
    kx: float,
) -> list[complex]:
    de[0] = e[1]
    de[1] = (kx**2.0 - perm.eps_o(z) * k0**2.0) * e[0]
    return de


def db_dz(
    db: list[complex],
    z: float,
    b: list[complex],
    perm: Permittivity,
    k0: float,
    kx: float,
) -> list[complex]:
    epsz = perm.eps_o(z)
    db[0] = b[1]
    db[1] = (
        perm.deriv_o(z) / epsz * b[1]
        + ((epsz / perm.eps_e(z)) * kx**2.0 - epsz * k0**2.0) * b[0]
    )
    return db


def match_bc_s(e1, de1):
    return e1, de1


def match_bc_p(eps2_o, eps1_o, b1, db1):
    b2 = b1
    db2 = eps2_o / eps1_o * db1
    return b2, db2


def match_bc(eps2_o, eps1_o, f1, df1, pol: Polarization):
    if pol is Polarization("s"):
        return match_bc_s(f1, df1)
    elif pol is Polarization("p"):
        return match_bc_p(eps2_o, eps1_o, f1, df1)
    else:
        raise InvalidPolarizationError(pol)


def refl_s(e, de, kz):
    if kz == 0:
        return 1.0 + 0j
    else:
        er = e - de / (1j * kz)
        ei = e + de / (1j * kz)
        return er / ei


def refl_p(b, db, eps_o, kz):
    if kz == 0:
        return 1.0 + 0j
    else:
        br = b - db / (1j * kz * eps_o)
        bi = b + db / (1j * kz * eps_o)
        return br / bi


def refl(f, df, eps_o, kz, pol: Polarization) -> complex:
    if pol is Polarization("s"):
        return refl_s(f, df, kz)
    elif pol is Polarization("p"):
        return refl_p(f, df, eps_o, kz)
    else:
        raise InvalidPolarizationError(pol)


def trans_s(e, de, kz):
    if kz == 0:
        return 0.0 + 0j
    else:
        return 1.0 / (0.5 * (e + de / (1j * kz)))


def trans_p(b, db, eps_o, kz):
    if kz == 0:
        return 0.0 + 0j
    else:
        return 1.0 / (0.5 * (b + db / (1j * kz * eps_o)))


def trans(f, df, kz, eps_o, pol: Polarization) -> complex:
    if pol is Polarization("s"):
        return trans_s(f, df, kz)
    elif pol is Polarization("p"):
        return trans_p(f, df, kz, eps_o)
    else:
        raise InvalidPolarizationError(pol)


def helmholtz_ode_func(
    eps: Permittivity, kx: float, k0: float, pol: Polarization
) -> OdeFunction:
    if pol == Polarization("s"):
        de = np.array([0.0 + 0j, 0.0 + 0j])
        return lambda z, e: de_dz(de, z, e, eps, k0, kx)  # type: ignore
    elif pol == Polarization("p"):
        db = np.array([0.0 + 0j, 0.0 + 0j])
        return lambda z, b: db_dz(db, z, b, eps, k0, kx)  # type: ignore
    else:
        raise InvalidPolarizationError(pol)


def integrate_helmholtz(
    f0: list[complex],
    x: list[float] | ArrayF64,
    eps: Permittivity,
    k0: float,
    kx: float,
    pol: Polarization,
    ode_options: dict = {},
) -> dict:
    ode_func = helmholtz_ode_func(eps, kx, k0, pol)
    x1, x2 = x[0], x[-1]
    out: dict = scipy.integrate.solve_ivp(
        ode_func, [x1, x2], f0, t_eval=x, **ode_options
    )
    return out


# A convenicence object. We calculate reflectivities and transmitivities
# based on the value on the right side of a boundary. It also gives the initial conditions
# for the Helmholtz equation, propagating a wave forward (which is more intuitive than the
# backward integration we use for solving the problem the first time).
@dataclass(frozen=True)
class HelmholtzInitialCondition:
    x: float
    f: complex
    df: complex
    eps_o: complex


@dataclass(frozen=True)
class StackSolution:
    stack: Stack
    ics: list[HelmholtzInitialCondition]
    kvec: KVector
    pol: Polarization

    def __post_init__(self):
        if not self.pol in list(Polarization):
            raise InvalidPolarizationError(self.pol)
        assert len(self.stack.layers) == len(self.ics)

    def kz(self, eps_o: complex, eps_e: None | complex = None) -> complex:
        eps_e = eps_o if eps_e is None else eps_e
        return self.kvec.kz(eps_o, eps_e, self.pol)

    @property
    def layers(self) -> list[Layer]:
        return self.stack.layers

    @property
    def r(self) -> complex:
        kz = self.kz(1.0 + 0j)
        v = self.ics[0]
        return refl(v.f, v.df, v.eps_o, kz, self.pol)

    @property
    def R(self) -> float:
        r = self.r
        return (r * r.conjugate()).real

    @property
    def t(self) -> complex:
        kz = self.kz(1.0 + 0j)
        v = self.ics[0]
        return trans(v.f, v.df, kz, v.eps_o, self.pol)

    @property
    def T(self) -> float:
        t = self.t
        kzi = self.kz(1.0 + 0j)
        permf = self.stack.last.perm
        zf = self.stack.length
        kzf = self.kz(permf.eps_o(zf), permf.eps_e(zf))
        if self.pol == Polarization("p"):
            return (
                0.0
                if kzi == 0
                else (t * t.conjugate()).real * np.real(kzf / kzi / permf.eps_o(zf))
            )
        else:
            return 0.0 if kzi == 0 else (t * t.conjugate()).real * np.real(kzf / kzi)

    def compute_field(self, layer_index: int, points: int):
        layer = self.layers[layer_index]
        v = self.ics[layer_index]
        x = np.linspace(v.x + layer.thickness, v.x, points)
        k = self.kvec
        out = integrate_helmholtz([v.f, v.df], x, layer.perm, k.k0, k.kx, self.pol)
        return x, out["y"]


def initial_condition(
    stack: Stack, kvec: KVector, pol: Polarization
) -> HelmholtzInitialCondition:
    x = stack.length
    eps_o = stack.last.perm.eps_o(x)
    eps_e = stack.last.perm.eps_e(x)
    f = 1.0 + 0j
    df = 0.0 + 1j * kvec.kz(eps_o, eps_e, pol)
    return HelmholtzInitialCondition(x=x, f=f, df=df, eps_o=eps_o)


def solve_stack(
    stack: Stack, om: float, th: float, pol: Polarization | str
) -> StackSolution:
    """Solves the helmholtz equation for an incident wave from vacuum through a stack."""
    kvec = KVector.create(om, th)
    pol = Polarization(pol)

    def loop(acc, layers):
        if len(layers) > 0:
            initial = acc[-1]
            layer = layers.pop(-1)
            x1 = initial.x
            x2 = x1 - layer.thickness
            f1, df1 = match_bc(
                layer.perm.eps_o(x1), initial.eps_o, initial.f, initial.df, pol
            )
            ode_options = {"max_step": min(layer.thickness / 100, 0.1 / kvec.k0)}
            out = integrate_helmholtz(
                [f1, df1],
                [x1, x2],
                layer.perm,
                kvec.k0,
                kvec.kx,
                pol,
                ode_options=ode_options,
            )
            f2, df2 = out["y"][:, -1]
            value = HelmholtzInitialCondition(x2, f2, df2, layer.perm.eps_o(x2))
            return loop(acc + [value], layers)
        else:
            out = list(acc)
            out.reverse()
            return out

    layers = list(stack.layers)  # careful! Make a copy, because we'll mutate the layers
    layers.pop(-1)  # remove last layer (sets initial condition)
    ics = loop([initial_condition(stack, kvec, pol)], layers)
    return StackSolution(stack=stack, ics=ics, kvec=kvec, pol=pol)


# -----------------------------------------------------------------------------
# Scripting


def main(plt) -> None:
    layers = [
        Layer.create(1e-5, lambda _: 1.5**2.0 + 0.01j),
        Layer.create(math.inf, lambda _: 1.5**2.0 + 0.0j),
    ]
    stack = Stack(layers)
    om = 2.35e15
    th = 0.0
    pol = Polarization("p")
    sol = solve_stack(stack, om, th, pol)
    print(f"R = {sol.R}")
    print(f"T = {sol.T}")
    print(f"R + T = {sol.R + sol.T}")

    x, y = sol.compute_field(0, 1_000)
    plt.figure()
    plt.plot(x, y[0, :])
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(plt)
