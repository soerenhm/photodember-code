# Fermi-Dirac integrals

The script `fdintegrals.jl` tabulates various Fermi-Dirac integrals for non-parabolic band structures: $\epsilon^k (1 + \alpha \epsilon^k) = \hbar^2 k^2 / 2m $. Running the julia script as is, generates the table with the same model parameters used in our manuscript.

# How to use

- Install [Julia](https://julialang.org/)
- Install the following packages:
  - QuadGK
  - CSV
  - JSON
  - DataFrames
- Run `fdintegrals.jl`.