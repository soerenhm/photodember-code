from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    c_0: float
    h: float
    hbar: float
    mu_0: float
    eps_0: float
    kB: float
    e: float
    m_e: float
    m_p: float


SI = PhysicalConstants(
    c_0 = 2.99792458e8,
    h = 6.62607015e-34,
    hbar = 1.054571817e-34,
    mu_0 = 1.25663706212e-6,
    eps_0 = 8.8541878128e-12,
    kB = 1.380649e-23,
    e = 1.602176634e-19,
    m_e = 9.1093837015e-31,
    m_p = 1.67262192369e-27
)