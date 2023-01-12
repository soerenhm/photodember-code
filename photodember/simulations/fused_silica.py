# %%

from __future__ import annotations
from dataclasses import replace
import pathlib
import json
import numpy as np

from photodember.simulations.core import run_simulation, SimulationConfig

# %%


def generate_excitation_data_simple_exp(
    surface_density: float,
    kinetic_energy_eV: float,
    length_scale_mu: float,
    x_mu: np.ndarray,
):
    data = np.zeros((len(x_mu), 3))
    data[:, 0] = np.copy(x_mu)
    data[:, 1] = surface_density * np.exp(-x_mu / length_scale_mu)
    data[:, 2] = kinetic_energy_eV * np.ones_like(x_mu)
    return data


length_scale_nm = 40
np.savetxt(
    f"photodember/data/Excitation FS SimpleExp {length_scale_nm} nm.txt",
    generate_excitation_data_simple_exp(
        1.45e28, 1.075, length_scale_nm * 1e-3, np.linspace(0, 2, 1000)
    ),
)

# %%

default_config = SimulationConfig(
    domain_size=1e-6,
    resolution=1e-9,
    initial_electron_hole_density=1e18,
    excitation_datafile=r"photodember\data\Excitation FS SimpleExp 40 nm.txt",
    excitation_time_fwhm=35e-15,
    excitation_time_zero=105e-15,
    fdtable_conduction_band="photodember/data/SiO2_CB.csv",
    fdtable_valence_band="photodember/data/SiO2_VB.csv",
    carrier_phonon_thermalization_time=1e-11,
    electron_hole_thermalization_time=1e-14,
    electron_mobility=3.5e-4,
    hole_mobility=5.8e-5,
    phonon_temperature=300.0,
    relative_permittivity=3.8,
)


# %%

mobility_scaling = [1.0]  # [0.1, 1.0, 10.0]
for mu_scaling in mobility_scaling:
    save_file = f"photodember/simulations/fs1_source_term_with_T_rise_1nm_35fs_muscale-{mu_scaling}_SimpleExp-{length_scale_nm}nm.dat"
    meta_file = save_file.replace(".dat", ".json")

    mu_e = default_config.electron_mobility * mu_scaling
    mu_h = default_config.hole_mobility * mu_scaling
    conf = replace(default_config, electron_mobility=mu_e, hole_mobility=mu_h)
    simul, initial_state = conf.create_simulation()
    pde = simul.create_pde()

    if not pathlib.Path(meta_file).exists():
        with open(save_file.replace(".dat", ".json"), "w") as io:
            json.dump(conf.to_dict(), io)
    times = np.linspace(0.0, 2.0e-12, 1000)  # , np.linspace(1.001e-12, 5e-12, 500))
    result = run_simulation(pde, save_file, initial_state, times)
