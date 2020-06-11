using ..Atmos
using ..Atmos: thermo_state
using ..Mesh.Topologies
using ..Mesh.Grids
using ..Thermodynamics
using LinearAlgebra

# Simple horizontal averages
function vars_atmos_refstate_perturbations(m::AtmosModel, FT)
    @vars begin
        ref_state::vars_atmos_refstate_perturbations(m.ref_state, FT)
    end
end
vars_atmos_refstate_perturbations(::ReferenceState, FT) = @vars()
function vars_atmos_refstate_perturbations(rs::HydrostaticState, FT)
    @vars begin
        rho::FT
        pres::FT
        temp::FT
        et::FT
        qt::FT
    end
end
num_atmos_refstate_perturbation_vars(m, FT) =
    varsize(vars_atmos_refstate_perturbations(m, FT))
atmos_refstate_perturbation_vars(m, array) =
    Vars{vars_atmos_refstate_perturbations(m, eltype(array))}(array)

function atmos_refstate_perturbations!(
    atmos::AtmosModel,
    state,
    aux,
    thermo,
    vars,
)
    atmos_refstate_perturbations!(
        atmos.ref_state,
        atmos,
        state,
        aux,
        thermo,
        vars,
    )
    return nothing
end
function atmos_refstate_perturbations!(
    ::ReferenceState,
    ::AtmosModel,
    state,
    aux,
    thermo,
    vars,
)
    return nothing
end
function atmos_refstate_perturbations!(
    rs::HydrostaticState,
    atmos::AtmosModel,
    state,
    aux,
    thermo,
    vars,
)
    vars.rho = state.ρ - rs.ρ
    vars.pres = thermo.pres - rs.p
    vars.temp = thermo.temp - rs.T
    vars.et = (state.ρe / state.ρ) - (rs.ρe / rs.ρ)
    # FIXME properly
    if atmos.moisture isa EquilMoist
        vars.qt = (thermo.moisture.ρq_tot / state.ρ) - (rs.ρq_tot / rs.ρ)
    else
        vars.qt = Inf
    end

    return nothing
end

function atmos_refstate_perturbations_init(dgngrp::DiagnosticsGroup, currtime)
    dg = Settings.dg
    atmos = dg.balance_law
    FT = eltype(Settings.Q)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    dims = OrderedDict(
        "points" => (collect(1:npoints), Dict()),
        "elements" => (collect(1:nrealelem), Dict()),
    )
    vars = OrderedDict()
    varnames = map(
        s -> startswith(s, "ref_state.") ? s[11:end] : s,
        flattenednames(vars_atmos_refstate_perturbations(atmos, FT)),
    )
    for varname in varnames
        vars[varname] = (("points", "elements"), FT, Dict())
    end

    dprefix = @sprintf(
        "%s_%s_%s_rank%04d",
        dgngrp.out_prefix,
        dgngrp.name,
        Settings.starttime,
        mpirank,
    )
    dfilename = joinpath(Settings.output_dir, dprefix)
    init_data(dgngrp.writer, dfilename, dims, vars)

    return nothing
end

"""
    atmos_refstate_perturbations_collect(dgngrp, currtime)

Perform a global grid traversal to compute various diagnostics.
"""
function atmos_refstate_perturbations_collect(dgngrp::DiagnosticsGroup, currtime)
    dg = Settings.dg
    atmos = dg.balance_law
    # FIXME properly
    if !isa(atmos.ref_state, HydrostaticState)
        @warn """
            Diagnostics $(dgngrp.name): has useful output only for `HydrostaticState`
            """
    end

    mpicomm = Settings.mpicomm
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)

    # extract grid information
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get needed arrays onto the CPU
    if array_device(Q) isa CPU
        state_array = Q.realdata
        aux_array = dg.state_auxiliary.realdata
    else
        state_array = Array(Q.realdata)
        aux_array = Array(dg.state_auxiliary.realdata)
    end
    FT = eltype(state_array)

    # Visit each node of the state variables array and:
    # - generate and store the thermo variables,
    # - record the perturbations
    #
    thermo_array =
        [zeros(FT, num_thermo(atmos, FT)) for _ in 1:npoints, _ in 1:nrealelem]
    perturbations_array = Array{FT}(
        undef,
        npoints,
        nrealelem,
        num_atmos_refstate_perturbation_vars(atmos, FT),
    )
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state_conservative(dg, state_array, ijk, e)
        aux = extract_state_auxiliary(dg, aux_array, ijk, e)

        thermo = thermo_vars(atmos, thermo_array[ijk, e])
        compute_thermo!(atmos, state, aux, thermo)

        perturbations = atmos_refstate_perturbation_vars(
            atmos,
            view(perturbations_array[ijk, e, :]),
        )
        atmos_refstate_perturbations!(atmos, state, aux, thermo, perturbations)
    end

    varvals = OrderedDict()
    varnames = map(
        s -> startswith(s, "ref_state.") ? s[11:end] : s,
        flattenednames(vars_atmos_refstate_perturbations(atmos, FT)),
    )
    for (vari, varname) in enumerate(varnames)
        varvals[varname] = perturbations_array[:, :, vari]
    end

    # write output
    append_data(dgngrp.writer, varvals, currtime)

    MPI.Barrier(mpicomm)
    return nothing
end # function collect

function atmos_refstate_perturbations_fini(dgngrp::DiagnosticsGroup, currtime) end
