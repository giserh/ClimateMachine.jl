using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid,
    VerticalDirection,
    HorizontalDirection,
    EveryDirection,
    min_node_distance
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: DGModel, init_ode_state, remainder_DGModel
using ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics:
    air_density,
    soundspeed_air,
    internal_energy,
    PhaseDry_given_pT,
    PhasePartition
using ClimateMachine.Atmos:
    AtmosModel,
    SphericalOrientation,
    DryModel,
    NoPrecipitation,
    NoRadiation,
    ConstantViscosityWithDivergence,
    vars_state_conservative,
    vars_state_auxiliary,
    Gravity,
    HydrostaticState,
    IsothermalProfile,
    AtmosAcousticGravityLinearModel,
    AtmosAcousticLinearModel,
    altitude,
    latitude,
    longitude,
    gravitational_potential
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = false

"""
    main()

Run this test problem
"""
function main()
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 5

    # Real test should be run for 33 hour, which is approximate time for the
    # pulse to go around the whole sphere
    # but for CI we only run 1 hour
    # timeend = 60 * 60 * 33
    timeend = 60 * 60

    # Do the output every hour
    outputtime = 60 * 60

    # Expected result is L2-norm of the final solution
    expected_result = Dict()
    expected_result[Float64, true] = 9.29934587510993e13
    expected_result[Float64, false] = 9.299357314714439e13

    @testset "acoustic wave" begin
        for FT in (Float64,)# Float32)
            for explicit in (true, false)
                result = run(
                    mpicomm,
                    polynomialorder,
                    numelem_horz,
                    numelem_vert,
                    timeend,
                    outputtime,
                    ArrayType,
                    FT,
                    explicit,
                )
                @test result ≈ expected_result[FT, explicit]
            end
        end
    end
end

"""
    run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        outputtime,
        ArrayType,
        FT,
        explicit_solve,
    )

Run the actual simulation.
"""
function run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
    explicit_solve,
)

    # Structure to pass around to setup the simulation
    setup = AcousticWaveSetup{FT}()

    # Create the cubed sphere mesh
    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + setup.domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = cubedshellwarp,
    )
    hmnd = min_node_distance(grid, HorizontalDirection())
    vmnd = min_node_distance(grid, VerticalDirection())

    T_profile = IsothermalProfile(param_set, setup.T_ref)

    # This is the base model which defines all the data (all other DGModels
    # for substepping components will piggy-back off of this models data)
    fullmodel = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = SphericalOrientation(),
        ref_state = HydrostaticState(T_profile),
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = setup,
    )
    dg = DGModel(
        fullmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )
    Q = init_ode_state(dg, FT(0))

    # The linear model which contains the fast modes
    # acousticmodel = AtmosAcousticLinearModel(fullmodel)
    acousticmodel = AtmosAcousticGravityLinearModel(fullmodel)

    vacoustic_dg = DGModel(
        acousticmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = dg.state_auxiliary,
    )

    # Advection model is the difference between the fullmodel and acousticmodel.
    # This will be handled with explicit substepping (time step in between the
    # vertical and horizontally acoustic models)
    rem_dg = remainder_DGModel(dg, (vacoustic_dg,))

    # determine the time step for the model components
    acoustic_speed = soundspeed_air(fullmodel.param_set, FT(setup.T_ref))
    advection_speed = 1 # What's a reasonable number here?

    vacoustic_dt = vmnd / acoustic_speed
    remainder_dt = min(min(hmnd, vmnd) / advection_speed, hmnd / acoustic_speed)
    odesolver = if explicit_solve
        remainder_dt = 5vacoustic_dt

        nsteps_output = ceil(outputtime / remainder_dt)
        remainder_dt = outputtime / nsteps_output
        nsteps = ceil(Int, timeend / remainder_dt)
        @assert nsteps * remainder_dt ≈ timeend

        vacoustic_solver =
            LSRK54CarpenterKennedy(vacoustic_dg, Q; dt = vacoustic_dt)
        rem_solver = MRIGARKERK45aSandu(
            rem_dg,
            vacoustic_solver,
            Q;
            dt = remainder_dt,
        )

        rem_solver
    else
        vacoustic_dt = 200vacoustic_dt
        element_size = (setup.domain_height / numelem_vert)

        nsteps_output = ceil(outputtime / vacoustic_dt)
        vacoustic_dt = outputtime / nsteps_output
        nsteps = ceil(Int, timeend / vacoustic_dt)
        @assert nsteps * vacoustic_dt ≈ timeend

        rem_solver = LSRK54CarpenterKennedy(rem_dg, Q; dt = remainder_dt)
        vacoustic_solver = MRIGARKESDIRK24LSA(
            vacoustic_dg,
            LinearBackwardEulerSolver(ManyColumnLU(); isadjustable = false),
            rem_solver,
            Q;
            dt = vacoustic_dt,
        )

        vacoustic_solver
    end

    # print some initial diagnostic information
    eng0 = norm(Q)
    @info @sprintf(
        """Starting
           ArrayType       = %s
           FT              = %s
           polynomialorder = %d
           numelem_horz    = %d
           numelem_vert    = %d
           acoustic dt     = %.16e
           remainder_dt    = %.16e
           norm(Q₀)        = %.16e
           """,
        "$ArrayType",
        "$FT",
        polynomialorder,
        numelem_horz,
        numelem_vert,
        vacoustic_dt,
        remainder_dt,
        eng0
    )

    # Setup the filtering callback
    filterorder = 18
    filter = ExponentialFilter(grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(1) do
        Filters.apply!(Q, 1:size(Q, 2), grid, filter, VerticalDirection())
        nothing
    end

    # Set up the information callback
    starttime = Ref(now())
    # cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
    cbinfo = EveryXSimulationSteps(nsteps_output) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo, cbfilter)

    # Setup the vtk callback
    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_acousticwave" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(ArrayType)_$(FT)" *
            "_$(explicit_solve ? "Explicit_Explicit" : "Implicit_Explicit")"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, fullmodel)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(nsteps_output) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, fullmodel)
        end
        callbacks = (callbacks..., cbvtk)
    end

    # Solve the ode
    solve!(
        Q,
        odesolver;
        numberofsteps = nsteps,
        adjustfinalstep = false,
        callbacks = callbacks,
    )

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(bl, state, aux, coords, t)
    # callable to set initial conditions
    FT = eltype(state)

    λ = longitude(bl, aux)
    φ = latitude(bl, aux)
    z = altitude(bl, aux)

    β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(setup.nv * FT(π) * z / setup.domain_height)
    Δp = setup.γ * f * g
    p = aux.ref_state.p + Δp

    ts = PhaseDry_given_pT(bl.param_set, p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)
    nothing
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "acousticwave",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
    auxnames = flattenednames(vars_state_auxiliary(model, eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
