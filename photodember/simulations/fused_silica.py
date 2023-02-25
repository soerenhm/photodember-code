# %%

from __future__ import annotations
from dataclasses import replace
import pathlib
import json
import numpy as np

from pathlib import Path
from photodember.simulations.core import (
    run_simulation,
    SimulationConfig,
)

# %%

fs_config = SimulationConfig(
    domain_size=1e-6,
    resolution=1e-9,
    initial_electron_hole_density=1e18,
    excitation_datafile=r"photodember\data\Excitation fused silica MRE 2.6 Jcm2.txt",
    excitation_time_fwhm=35e-15,
    excitation_time_zero=105e-15,
    fdtable_conduction_band=r"photodember\data\SiO2_alpha-2.0_CB.csv",
    fdtable_valence_band=r"photodember\data\SiO2_alpha-2.0_VB.csv",
    carrier_phonon_thermalization_time=1e-11,
    electron_hole_thermalization_time=1e-14,
    electron_mobility=4.22e-2,
    hole_mobility=2.11e-4,
    phonon_temperature=300.0,
    relative_permittivity=3.8,
)

# %% Main simulation

simul, initial_state = fs_config.create_simulation()
pde = simul.create_pde()
save_file = f"photodember/simulations/SiO2_fluence_tau-120fs.dat"
meta_file = save_file.replace(".dat", ".json")
if not pathlib.Path(meta_file).exists():
    with open(save_file.replace(".dat", ".json"), "w") as io:
        json.dump(fs_config.to_dict(), io)
times = np.linspace(0.0, 2.0e-12, 2000)
result = run_simulation(pde, save_file, initial_state, times)


# %% Simulate THz scan (warning: takes a long time!)

data_dir = Path(r"photodember\data\thz-scan")
excitation_data = []
for filepath in data_dir.glob("*.txt"):
    fluence = float(filepath.name.split("MRE ")[1].split(" Jcm2")[0])
    excitation_data.append((fluence, filepath))

for exc_data in excitation_data:
    F, excitation_datafile = exc_data
    save_file = f"photodember/simulations/thz-scan/SiO2_fluence-{F}-Jm-2.dat"
    meta_file = save_file.replace(".dat", ".json")
    excitation_data = np.loadtxt(excitation_datafile)
    excitation_data[:, 0] *= 1e6  # z should be in um, but it's not
    np.savetxt("excitation_data.tmp.txt", excitation_data)
    mu_e = fs_config.electron_mobility
    mu_h = fs_config.hole_mobility
    conf = replace(
        fs_config,
        electron_mobility=mu_e,
        hole_mobility=mu_h,
        excitation_datafile="excitation_data.tmp.txt",
    )
    simul, initial_state = conf.create_simulation()
    pde = simul.create_pde()

    if not pathlib.Path(meta_file).exists():
        with open(save_file.replace(".dat", ".json"), "w") as io:
            json.dump(conf.to_dict(), io)
    times = np.linspace(0.0, 2.0e-12, 2000)  # , np.linspace(1.001e-12, 5e-12, 500))
    result = run_simulation(pde, save_file, initial_state, times)
