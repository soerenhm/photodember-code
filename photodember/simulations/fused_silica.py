# %% 

from __future__ import annotations
import pathlib
import json
import numpy as np

from photodember.simulations.dielectric import DielectricConfig
from photodember.simulations.simulator import run_simulation
# from photodember.src.constants import SI
# from photodember.src.transport.simulation import *


# -----------------------------------------------------------------------------
# Test simulation....

def main():
    conf = DielectricConfig(
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
    return result

# %%

conf = DielectricConfig(
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

# # %%

# save_file = "photodember/simulations/fs1_source_term_with_T_rise_2.dat"
# meta_file = save_file.replace(".dat", ".json")
# if not pathlib.Path(meta_file).exists():
#     with open(save_file.replace(".dat", ".json"), "w") as io:
#         json.dump(conf.to_dict(), io)
# times = np.linspace(0.0, 1e-12, 1000)
# result = run_simulation(pde, save_file, initial_state, times)


# # %%
# import matplotlib.pyplot as plt

# tfwhm = 50e-15
# t, dt = np.linspace(-3*tfwhm, 3*tfwhm, 1000, retstep=True)
# r = lambda t: .5 * (1 + erf(np.sqrt(4*np.log(2))*t/tfwhm))
# A = np.sqrt(4*np.log(2)/np.pi)/tfwhm
# drdt = lambda t: A * np.exp(-4*np.log(2)*(t/tfwhm)**2)
# N = 1e28 * r(t)
# T = (7000-300.0) * r(t) + 300.0

# electron = simul.particles[0]

# def dHdt(Ti, Tf, Nf, t, T, eta):
#     dN = Nf * drdt(t)
#     dT = (Tf - Ti) * drdt(t)
#     U, N = electron.energy_density, electron.number_density
#     dH = (U.dT(T, eta) - U.deta(T, eta) * N.dT(T, eta)/N.deta(T, eta)) * dT + U.deta(T, eta)/N.deta(T, eta) * dN
#     return dH

# eta = [solve_reduced_chemical_potential(electron.number_density, Ti, Ni, [-100, 100]) for Ti, Ni in zip(T, N)]
# dH = dHdt(300.0, 7000.0, 1e28, t, T, eta)

# G = lambda t: 1e28 * drdt(t)
# H = lambda t: electron.number_density

# # %%

# Nt = electron.number_density(T, eta)
# Ht = electron.energy_density(T, eta)

# # plt.plot(t, Nt, "b-")
# # plt.twinx()
# plt.plot(t, Ht, "r-")
# plt.plot(t, np.cumsum(dH*dt))

# # %%
# E = np.array([s.electric_field for s in result])
# Ne = np.array([s.number_density[0] for s in result])
# Nh = np.array([s.number_density[1] for s in result])
# Te = np.array([s.temperature[0] for s in result])
# Th = np.array([s.temperature[1] for s in result])
# # %%

# import matplotlib.pyplot as plt
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# plt.rcParams["font.family"] = "Arial"
# plt.rcParams["axes.labelsize"] = "large"
# plt.rcParams["xtick.direction"] = "in"
# plt.rcParams["ytick.direction"] = "in"

# # %%

# plt.figure(figsize=(5,5.5))
# t = np.array(times) * 1e15

# plt.subplot(2,1,1)
# plt.plot(t, Te[:,0], "r-", label="electron")
# plt.plot(t, Th[:,0], "b-", label="hole")
# # plt.plot(t, Te[:,-1], "r--")#, label="electron")
# # plt.plot(t, Th[:,-1], "b--")#, label="hole")
# plt.xlim([0, 1e3])
# plt.gca().yaxis.set_major_locator(MultipleLocator(300))
# plt.gca().xaxis.set_major_locator(MultipleLocator(100))
# plt.legend(frameon=False)
# plt.ylim([300, 3300])
# plt.ylabel("Surface temperature (K)")

# plt.subplot(2,1,2)
# plt.plot(t, E[:,1]*.5*1e-7, "g-")
# plt.xlim([0, 1e3])
# plt.gca().xaxis.set_major_locator(MultipleLocator(100))
# plt.xlabel("Time (fs)")
# plt.ylabel(r"Surface field (10$^7$ V/m)")
# plt.ylim([0, 1.6])

# plt.tight_layout()
# plt.savefig("Temperature.png", dpi=300, facecolor="w")
# # %%

# %%
