from dataclasses import dataclass, replace
import numpy as np
from numpy.typing import NDArray
from typing import List, Union

from photodember.src.constants import SI


## --------------------------------------------------------------------------------
## Types and globals

ArrayF64 = NDArray[np.float64]


## --------------------------------------------------------------------------------
## Model definitions

@dataclass(frozen=True)
class Lorentz:
    """Lorentz oscillator model. Values are in eV."""
    F: float
    gam: float
    om0: float
    alpha: float

    def value(self, om, shift=0.0):
        alpha, om0, gam, F = self.alpha, self.om0+shift, self.gam, self.F
        gam = gam * np.exp(-alpha*((om-om0)/gam)**2)
        denom = 1.0/((om**2-om0**2)**2 + (om*gam)**2)
        eps1 = F*(om0**2-om**2)*denom
        eps2 = F*om*gam*denom
        return eps1 + 1j*eps2

    def dielectric_function(self, om, shift=0.0):
        return self.value(om * SI.hbar/SI.e, shift=shift/SI.e)

    @staticmethod
    def shift_bandgap(model: "Lorentz", shift: float) -> "Lorentz":
        return replace(model, om0 = model.om0 + shift)


@dataclass(frozen=True)
class MultiLorentz:
    eps_inf: float
    oscillators: List[Lorentz]

    def dielectric_function(self, om, shift=0.0):
        eps = self.eps_inf + 0j
        return eps + sum([t.dielectric_function(om, shift=shift) for t in self.oscillators])

    @staticmethod
    def shift_bandgap(model: "MultiLorentz", shift: float) -> "MultiLorentz":
        return replace(model, oscillators = [Lorentz.shift_bandgap(osc, shift) for osc in model.oscillators])


@dataclass(frozen=True)
class Drude:
    eps_val: float
    n_val: float
    gam: float
    m: float

    def dielectric_function(self, om: float, n: float) -> complex:
        m, n_val, gam, eps_val = self.m, self.n_val, self.gam, self.eps_val
        om_p = np.sqrt( SI.e**2 * n / (SI.eps_0 * m) )
        return eps_val + (1 - eps_val)*n/n_val - (om_p/om)*(om_p/(om + 1j*gam))


@dataclass(frozen=True)
class DrudeLorentz:
    lorentz: MultiLorentz
    n_val: float
    gam: float
    m: float
    
    def dielectric_function(self, om: float, n: Union[float, ArrayF64], shift: Union[float, ArrayF64] = 0.0):
        m, n_val, gam = self.m, self.n_val, self.gam
        eps_rel = self.lorentz.dielectric_function(om, shift) # type: ignore
        om_p = np.sqrt( SI.e**2 * n / (SI.eps_0 * m) )
        return eps_rel + (1 - eps_rel)*n/n_val - om_p**2/(om**2 + 1j*om*gam)

    @staticmethod
    def shift_bandgap(model: "DrudeLorentz", shift: float) -> "DrudeLorentz":
        lorentz = model.lorentz
        return replace(model, lorentz=lorentz.shift_bandgap(lorentz, shift))


## --------------------------------------------------------------------------------
## Concrete models

sapphire = MultiLorentz(
    eps_inf=0.173,
    oscillators=[
        Lorentz(0.600, 0.093, 9.168, 0.221),
        Lorentz(2.390e2, 4.110, 18.113, 3.510),
        Lorentz(0.036e2, 0.514, 9.386, 1.924),
        Lorentz(0.754e2, 1.597, 13.384, 0.160),
        Lorentz(1.549e2, 2.639, 15.141, 0.488),
        Lorentz(0.884e2, 2.690, 12.297, 0.638),
        Lorentz(2.335e2, 3.920, 21.394, 14.861),
        Lorentz(0.329e2, 3.093, 19.062, 6.922)
    ]
)


## --------------------------------------------------------------------------------
## Scripting

def main() -> None:
    model = sapphire
    ph = np.arange(0.5, 15, 0.01)
    oms = ph * SI.e / SI.hbar
    es = np.array([model.dielectric_function(om, 0.0) for om in oms])
    es2 = np.array([model.dielectric_function(om, -2*SI.e) for om in oms])
    plt.figure(dpi=150)
    plt.plot(ph, es.real, "m-", label="real")
    plt.plot(ph, es2.real, "m--")
    plt.plot(ph, es.imag, "c-", label="imag")
    plt.plot(ph, es2.imag, "c--")
    plt.legend(frameon=False)
    plt.xlabel("$\hbar \omega$ (eV)", fontsize="large")
    plt.ylabel("Dielectric function", fontsize="large")
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()