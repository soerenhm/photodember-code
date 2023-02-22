# Photo-Dember code

Python code for simulating the photo-Dember effect in highly-excited dielectrics. To use the code

# How to use

You first need to generate a table of the required Fermi-Dirac integrals by running the `fdintegrals.jl` file located in the `fdintegrals` folder of this repository.

To run the Python code, you need to ensure that you have all the dependencies (the essentials are: `numpy`, `scipy`, `pandas`, and `numba`). To facilitate this, you can create a conda environment from the `environments/conda.yaml` file.

The file `photodember/fused_silica.py` runs all the simulations. After that, you should be able to run all the other files in the `photodember` folder to generate the figures displayed in our manuscript.

# Warning

I've tried to optimize the code as best I can, but it's still very slow! The main bottleneck is the interpolations of the Fermi-Dirac integrals.