using QuadGK
using CSV, JSON, DataFrames

const e = 1.60217663e-19
const kB = 1.380649e-23
const me = 9.1093837e-31
const ħ = 1.05457182e-34
const T0 = 300.0
const kBT0 = kB * 300.0

"""
Fermi-dirac distribution. Arguments are expressed in units of energy over kB T. 
"""
function fermi_dirac(η, x)
    return 1.0 / (1. + exp(x - η))
end

"""
Scaled density of states for a non-parabolic band with dispersion
    ``ħk^2/2m = ϵ(1+αϵ)``.
Set `α = 0` for parabolic bands. Multiply the result by `density_of_states_scale(m)` to 
express it in SI units.
"""
function density_of_states(α, θ, x)
    u = θ * x
    return θ * sqrt(u) * (1. + 2*α*u) * sqrt(1. + α*u)
end

function density_of_states_scale(m)
    sqrt(2.0)/π^2 * (m*kBT0/ħ^2)^(3/2)
end

module Integrand
    using Main: kBT0, ħ, fermi_dirac, density_of_states, density_of_states_scale

    function number_density(dos, θ, η, x)
        return dos(θ, x) * fermi_dirac(η, x) 
    end

    function number_density_scale(m)
        return density_of_states_scale(m)
    end

    function energy_density(dos, θ, η, x)
        return dos(θ, x) * (θ*x) * fermi_dirac(η, x)
    end
    
    function energy_density_scale(m)
        return kBT0 * density_of_states_scale(m)
    end

    function kinetic_integral_j(j, α, θ, η, x)
        j = float(j)
        u = θ * x
        f = fermi_dirac(η, x)
        dos = density_of_states(α, θ, x)
        return dos * (f*(1-f)) * (1+α*u)/(1+2α*u)^2 * u^j * x
    end
    
    function kinetic_integral_j_scale(j, τ, m)
        j = float(j)
        τ/(3.0*m*π^2)*(2.0*m*kBT0/ħ^2)^(3/2) * (kBT0)^j
    end
    
    kinetic_integral_0(α, θ, η, x) = kinetic_integral_j(0, α, θ, η, x)
    kinetic_integral_1(α, θ, η, x) = kinetic_integral_j(1, α, θ, η, x)
    kinetic_integral_2(α, θ, η, x) = kinetic_integral_j(2, α, θ, η, x)
end

"""
Transform an integral over a function `f` on the interval `[0, ∞]` to the unit line
using variable substitution `t = x/(1-x)`.
"""
function transform_integrand(f)
    x -> begin
        a = 1.0/(1.0-x)
        t = x*a
        f(t) * a^2
    end
end

create_dos(α) = (θ, x) -> density_of_states(α, θ, x)

"""
Integrates over the energy in the CB and VB.
"""
function fermi_integral(f, η)
    if η <= 51.0
        return quadgk(transform_integrand(f), 0.0, 1.0, rtol=1e-15)
    else
        # Don't transform the integrand; it becomes very spiky at t ~ 1.0
        # The factor f(1 - f) has decayed to 1e-20 away from the center
        return quadgk(f, 0., η+50.0, rtol=1e-15)
    end
end

number_density(α, m, θ, η) = fermi_integral(x -> Integrand.number_density(create_dos(α), θ, η, x), η) .* Integrand.number_density_scale(m)
energy_density(α, m, θ, η) = fermi_integral(x -> Integrand.energy_density(create_dos(α), θ, η, x), η) .* Integrand.energy_density_scale(m)
kinetic_integral_0(τ, α, m, θ, η) = fermi_integral(x -> Integrand.kinetic_integral_0(α, θ, η, x), η) .* Integrand.kinetic_integral_j_scale(0, τ, m)
kinetic_integral_1(τ, α, m, θ, η) = fermi_integral(x -> Integrand.kinetic_integral_1(α, θ, η, x), η) .* Integrand.kinetic_integral_j_scale(1, τ, m)
kinetic_integral_2(τ, α, m, θ, η) = fermi_integral(x -> Integrand.kinetic_integral_2(α, θ, η, x), η) .* Integrand.kinetic_integral_j_scale(2, τ, m)


struct ScanParams{T}
    relative_mass::T
    non_parabolicity::T
    relaxation_time::T
    kBT0::T
end

function scan_fermi_integrals(params::ScanParams, θ, η, rtol=1e-12)
    args = [(θi, ηj) for ηj in η, θi in θ]
    α = params.non_parabolicity
    m = params.relative_mass * me
    τ = params.relaxation_time
    kBT0 = params.kBT0
    columns = [ "Reduced temperature",
                "Reduced chemical potential",
                "Temperature [K]", 
                "Chemical potential [J]", 
                "Number density [1/m^3]",
                "Energy density [J/m^3]",
                "I0 [s/(kg m^3)]",
                "I1 [J s/(kg m^3)]",
                "I2 [(J^2 s/(kg m^3)]"]
    function check_and_return_result(θ, η, quadgk_result)
        value, error = quadgk_result
        e = error / value
        if e > rtol
            @warn "Warning: Integral at (θ, η) = ($θ, $η) has unusually large relative error: $e"
        end
        return value
    end
    output = zeros(length(args), length(columns))
    for i in eachindex(args)
        arg = args[i]
        if i % 1000 == 0 
            println("Progress: $(i/length(args))")
        end
        θ, η = arg
        T = θ * (kBT0/kB)
        μ = η * kB * T
        check(arg) = check_and_return_result(θ, η, arg)
        N = number_density(α, m, θ, η) |> check
        U = energy_density(α, m, θ, η) |> check
        I0 = kinetic_integral_0(τ, α, m, θ, η) |> check
        I1 = kinetic_integral_1(τ, α, m, θ, η) |> check
        I2 = kinetic_integral_2(τ, α, m, θ, η) |> check
        output[i, :] = [θ, η, T, μ, N, U, I0, I1, I2]
    end
    return output, columns
end

function save_scan(filename, params, scan_output)
    output_matrix, column_names = scan_output
    df = DataFrame(output_matrix, :auto)
    meta = Dict(
        "relativeMass" => params.relative_mass,
        "nonParabolicity" => params.non_parabolicity,
        "relaxationTime" => params.relaxation_time,
    )
    CSV.write(filename, df, header=column_names, delim='\t')
    open(replace(filename, ".csv" => ".json"), "w") do io
        println(meta)
        JSON.print(io, meta)
    end
end


"""
Analytical test results for a non-degenerate semiconductor with parabolic bands.
"""
module Analytical
    using Main: kBT0, ħ

    function number_density(m, θ, η)
        return 1.0/4.0*(2.0*m*kBT0*θ/π/ħ^2)^(3/2) * exp(η)
    end

    function energy_density(m, θ, η)
        return (3/8*kBT0*θ) * (2.0*m*kBT0*θ/π/ħ^2)^(3/2) * exp(η)
    end

    function kinetic_integral_0(τ, m, θ, η)
        return τ * number_density(m, θ, η) / m
    end

    function kinetic_integral_1(τ, m, θ, η)
        return 5/2 * kBT0*θ * τ * number_density(m, θ, η) / m
    end

    function kinetic_integral_2(τ, m, θ, η)
        return 35/4 * (kBT0*θ)^2 * τ * number_density(m, θ, η) / m
    end
end


module Tests
    import Main
    import Main: Analytical
    using PyPlot; pygui(true)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = "large"

    function run_tests(θ)
        T = θ * Main.kBT0/Main.kB
        η = range(-200, 200, 400)
        α = 0.0
        m = 0.5*Main.me
        τ = 1e-12

        Ne_calculated = map(η -> Main.number_density(α, m, θ, η)[1], η)
        Ue_calculated = map(η -> Main.energy_density(α, m, θ, η)[1], η)
        Ie0_calculated = map(η -> Main.kinetic_integral_0(τ, α, m, θ, η)[1], η)
        Ie1_calculated = map(η -> Main.kinetic_integral_1(τ, α, m, θ, η)[1], η)
        Ie2_calculated = map(η -> Main.kinetic_integral_2(τ, α, m, θ, η)[1], η)

        Ne_analytical = map(η -> Analytical.number_density(m, θ, η), η)
        Ue_analytical = map(η -> Analytical.energy_density(m, θ, η), η)
        Ie0_analytical = map(η -> Analytical.kinetic_integral_0(τ, m, θ, η), η)
        Ie1_analytical = map(η -> Analytical.kinetic_integral_1(τ, m, θ, η), η)
        Ie2_analytical = map(η -> Analytical.kinetic_integral_2(τ, m, θ, η), η)

        relerror = (x, y) -> @. abs(x-y)/y

        plt.figure(figsize=(5.5,4), dpi=125)
        plt.title("T = $(round(Int, T)) K.")
        plt.semilogy(η, relerror(Ne_calculated, Ne_analytical), "b-", lw=2, alpha=0.75, label=L"$N_e$")
        plt.semilogy(η, relerror(Ue_calculated, Ue_analytical), "r-", lw=2, alpha=0.75, label=L"$U_e$")
        plt.semilogy(η, relerror(Ie0_calculated, Ie0_analytical), "m-", lw=2, alpha=0.75, label=L"$I_e^0$")
        plt.semilogy(η, relerror(Ie1_calculated, Ie1_analytical), "g-", lw=2, alpha=0.75, label=L"$I_e^1$")
        plt.semilogy(η, relerror(Ie2_calculated, Ie2_analytical), "y-", lw=2, alpha=0.75, label=L"$I_e^2$")
        plt.xlabel(L"Reduced Fermi energy, $η$")
        plt.ylabel("Relative error")
        plt.legend(frameon=false)
        plt.xlim(extrema(η))
        plt.tight_layout()
    end

    function plot_all(θ)
        T = θ * Main.kBT0/Main.kB
        η = range(-200, 200, 400)
        α = 0.0
        m = 0.5*Main.me
        τ = 1e-12

        Ne_calculated = map(η -> Main.number_density(α, m, θ, η)[1], η)
        Ue_calculated = map(η -> Main.energy_density(α, m, θ, η)[1], η)
        Ie0_calculated = map(η -> Main.kinetic_integral_0(τ, α, m, θ, η)[1], η)
        Ie1_calculated = map(η -> Main.kinetic_integral_1(τ, α, m, θ, η)[1], η)
        Ie2_calculated = map(η -> Main.kinetic_integral_2(τ, α, m, θ, η)[1], η)

        normalize = x -> x / maximum(abs.(x))

        plt.figure(figsize=(5.5,4), dpi=125)
        plt.title("T = $(round(Int, T)) K.")
        plt.semilogy(η, normalize(Ne_calculated), "b-", lw=2, alpha=0.75, label=L"$N_e$")
        plt.semilogy(η, normalize(Ue_calculated), "r-", lw=2, alpha=0.75, label=L"$U_e$")
        plt.semilogy(η, normalize(Ie0_calculated), "m-", lw=2, alpha=0.75, label=L"$I_e^0$")
        plt.semilogy(η, normalize(Ie1_calculated), "g-", lw=2, alpha=0.75, label=L"$I_e^1$")
        plt.semilogy(η, normalize(Ie2_calculated), "y-", lw=2, alpha=0.75, label=L"$I_e^2$")
        plt.xlabel(L"Reduced Fermi energy, $η$")
        plt.ylabel("Relative error")
        plt.legend(frameon=false)
        plt.xlim(extrema(η))
        plt.tight_layout()
    end
end

module FusedSilica
    import Main: me, e, kB, kBT0, save_scan, scan_fermi_integrals

    function run_scan()
        bandgap_eV = 8.9
        non_parabolicity = 2.0 / e # 1/J
        relmass_cb = 0.5
        relmass_vb = 3.0
        mobility_cb = 0.1 # m^2/V s
        relax_time_cb = mobility_cb * (relmass_cb * me) / e
        relax_time_vb = relax_time_cb

        npoints = 400
        Tmin = 250.0
        Tmax = 30_000.0
        eFmin = -0.6*bandgap_eV
        eFmax = 1.0*bandgap_eV

        α = non_parabolicity * kBT0 # dimensionless non-parabolicity
        params_cb = Main.ScanParams(relmass_cb, α, relax_time_cb, kBT0)
        params_vb = Main.ScanParams(relmass_vb, α, relax_time_vb, kBT0)
        T = range(1/Tmax, 1/Tmin, npoints) .|> inv |> reverse
        θ = kB * T ./ kBT0
        η = range(e*eFmin/(kB*Tmin), e*eFmax/(kB*Tmin), npoints)
        save_scan("photodember/data/SiO2_alpha-2.0_CB.csv", params_cb, scan_fermi_integrals(params_cb, θ, η))
        save_scan("photodember/data/SiO2_alpha-2.0_VB.csv", params_vb, scan_fermi_integrals(params_vb, θ, η))
    end

    function run_scan_linear_T()
        bandgap_eV = 8.9
        non_parabolicity = 0.4
        relmass_cb = 0.5
        relmass_vb = 3.0
        mobility_cb = 0.1 # m^2/V s
        relax_time_cb = mobility_cb * (relmass_cb * me) / e
        relax_time_vb = relax_time_cb

        npoints = 400
        Tmin = 200.0
        Tmax = 30_000.0
        eFmin = -0.6*bandgap_eV
        eFmax = 1.0*bandgap_eV

        params_cb = Main.ScanParams(relmass_cb, non_parabolicity, relax_time_cb, kBT0)
        params_vb = Main.ScanParams(relmass_vb, non_parabolicity, relax_time_vb, kBT0)
        T = range(Tmin, Tmax, npoints)
        θ = kB * T ./ kBT0
        η = range(e*eFmin/(kB*Tmin), e*eFmax/(kB*Tmin), npoints)
        save_scan("data/SiO2_reg_T_CB.csv", params_cb, scan_fermi_integrals(params_cb, θ, η))
        save_scan("data/SiO2_reg_T_VB.csv", params_vb, scan_fermi_integrals(params_vb, θ, η))
    end
end

@time FusedSilica.run_scan()
# FusedSilica.run_scan_linear_T()

# f = x -> Integrand.kinetic_integral_2(0.0, 1.0, 100.0, x)
# g = transform_integrand(f)
# t = range(0, 1, 1000)
# plt.plot(t, g.(t))
# Tests.run_tests(100.0)
# FusedSilica.run_scan()