#},

using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function init_agnesi_hs_lin!(bl, state, aux, (x, y, z), t)

    FT = eltype(state)

    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    c::FT = c_v / R_gas
    c2::FT = R_gas / c_p

    Tiso::FT = 250.0
    θ0::FT = Tiso

    Brunt::FT = _grav / sqrt(c_p * Tiso)
    Brunt2::FT = Brunt * Brunt
    g2::FT = _grav * _grav

    π_exner::FT = exp(-_grav * z / (c_p * Tiso))
    θ::FT = θ0 * exp(Brunt2 * z / _grav)
    ρ::FT = p0 / (R_gas * θ) * (π_exner)^c

    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)

    #initial velocity
    u = FT(20.0)

    #State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)# potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    state.ρ = ρ
    state.ρu = SVector{3, FT}(ρ * u, 0, 0)
    state.ρe = ρe_tot
end

function setmax(f, xmax, ymax, zmax)
    function setmaxima(xin, yin, zin)
        return f(xin, yin, zin; xmax = xmax, ymax = ymax, zmax = zmax)
    end
    return setmaxima
end

function warp_agnesi(xin, yin, zin; xmax = 1000.0, ymax = 1000.0, zmax = 1000.0)

    FT = eltype(xin)

    ac = FT(10000)
    hm = FT(1)
    xc = FT(0.5) * xmax
    zdiff = hm / (FT(1) + ((xin - xc) / ac)^2)

    x, y, z = xin, yin, zin + zdiff * (zmax - zin) / zmax
    return x, y, z
end

function config_agnesi_hs_lin(FT, N, resolution, xmax, ymax, zmax)

    u_relaxation = SVector(FT(20), FT(0), FT(0))
    c_sponge = 1
    amp = 1.0

    zsponge = FT(20000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)

    ##SR LSRK144
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    source = (Gravity(), rayleigh_sponge)

    #temp_profile_ref = DecayingTemperatureProfile{FT}(param_set)
    T_virt = FT(250)
    temp_profile_ref = IsothermalProfile(param_set, T_virt)
    ref_state = HydrostaticState(temp_profile_ref)
    nothing # hide

    _C_smag = FT(0.21)
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = Vreman(_C_smag),
        moisture = DryModel(),
        source = source,
        tracers = NoTracers(),
        init_state_conservative = init_agnesi_hs_lin!,
        ref_state = ref_state,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "Agnesi_HS_LINEAR",      # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        xmax,                    # Domain maximum size [m]
        ymax,                    # Domain maximum size [m]
        zmax,                    # Domain maximum size [m]
        param_set,               # Parameter set.
        init_agnesi_hs_lin!,     # Function specifying initial condition
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
        meshwarp = setmax(warp_agnesi, xmax, ymax, zmax),
    )

    return config
end

function main()

    FT = Float64

    N = 4
    Δh = FT(500)
    Δv = FT(240)
    resolution = (Δh, Δh, Δv)
    xmax = FT(244000)
    ymax = FT(2000)
    zmax = FT(35000)
    t0 = FT(0)
    timeend =  FT(15000) #FT(hrs * 60 * 60)

    Courant = FT(0.2)

    driver_config = config_agnesi_hs_lin(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = Courant,
    )

    #cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)

    #end

    #Exponential filter:
#=    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        @views begin
            solver_config.Q.data[:, 2, :] .-= 20*solver_config.Q[:,1,:]
            Filters.apply!(
                solver_config.Q,
                2:4,
                solver_config.dg.grid,
                filter,
            )
            solver_config.Q.data[:, 2, :] .+= 20*solver_config.Q[:,1,:]
        end
        nothing
    end
=#
    #End exponential filter

    result = ClimateMachine.invoke!(
        solver_config;
        #user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

