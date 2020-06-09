using MPI
using Distributions
using NCDatasets
using OrderedCollections
using Plots
using StaticArrays
using LinearAlgebra: Diagonal

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils

import ClimateMachine.DGMethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!

FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "docs", "plothelpers.jl"));

Base.@kwdef struct BurgersEquation{FT} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
    "Heat capacity"
    c::FT = 1
    "Vertical dynamic viscosity"
    μv::FT = 1e-4
    "Horizontal dynamic viscosity"
    μh::FT = 1e-2
    "Thermal diffusivity"
    α::FT = 0.01
    "IC Gaussian noise standard deviation"
    σ::FT = 1e-1
    "Rayleigh damping"
    γ::FT = μh / 0.08 / 1e-2 / 1e-2
    "Domain height"
    zmax::FT = 1
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α∇ρcT) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
end

m = BurgersEquation{FT}();

vars_state_auxiliary(::BurgersEquation, FT) = @vars(z::FT, T::FT);

vars_state_conservative(::BurgersEquation, FT) =
    @vars(ρ::FT, ρu::SVector{3, FT}, ρcT::FT);

vars_state_gradient(::BurgersEquation, FT) = @vars(u::SVector{3, FT}, ρcT::FT);

vars_state_gradient_flux(::BurgersEquation, FT) =
    @vars(μ∇u::SMatrix{3, 3, FT, 9}, α∇ρcT::SVector{3, FT});

function init_state_auxiliary!(
    m::BurgersEquation,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

function init_state_conservative!(
    m::BurgersEquation,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    z = aux.z
    ε1 = rand(Normal(0, m.σ))
    ε2 = rand(Normal(0, m.σ))
    state.ρ = 1
    ρu = 1 - 4 * (z - m.zmax / 2)^2 + ε1
    ρv = 1 - 4 * (z - m.zmax / 2)^2 + ε2
    ρw = 0
    state.ρu = SVector(ρu, ρv, ρw)

    state.ρcT = state.ρ * m.c * aux.T
end;

function update_auxiliary_state!(
    dg::DGModel,
    m::BurgersEquation,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(heat_eq_nodal_update_aux!, dg, m, Q, t, elems)
    return true # TODO: remove return true
end;

function heat_eq_nodal_update_aux!(
    m::BurgersEquation,
    state::Vars,
    aux::Vars,
    t::Real,
)
    aux.T = state.ρcT / (state.ρ * m.c)
end;

function compute_gradient_argument!(
    m::BurgersEquation,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρcT = state.ρcT
    transform.u = state.ρu / state.ρ
end;

function compute_gradient_flux!(
    m::BurgersEquation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.α∇ρcT = m.α * ∇transform.ρcT
    diffusive.μ∇u = Diagonal(SVector(m.μh, m.μh, m.μv)) * ∇transform.u
end;

function source!(
    m::BurgersEquation{FT},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    args...,
) where {FT}
    ẑ = SVector{3, FT}(0, 0, 1)
    ρ̄ū =
        state.ρ * SVector{3, FT}(
            0.5 - 2 * (aux.z - m.zmax / 2)^2,
            0.5 - 2 * (aux.z - m.zmax / 2)^2,
            0.0,
        )
    ρu_p = state.ρu - ρ̄ū
    source.ρu -= m.γ * (ρu_p - ẑ' * ρu_p * ẑ)
end;

function flux_first_order!(
    m::BurgersEquation,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρ = state.ρu

    u = state.ρu / state.ρ
    flux.ρu = state.ρu * u'
    flux.ρcT = u * state.ρcT
end;

function flux_second_order!(
    m::BurgersEquation,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρcT -= diffusive.α∇ρcT
    flux.ρu -= diffusive.μ∇u
end;

function boundary_state!(
    nf,
    m::BurgersEquation,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # bottom
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        state⁺.ρcT = state⁺.ρ * m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
    end
end;

function boundary_state!(
    nf,
    m::BurgersEquation,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # bottom
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        state⁺.ρcT = state⁺.ρ * m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        diff⁺.α∇ρcT = -n⁻ * m.flux_top
    end
end;

N_poly = 5;

nelem_vert = 20;

zmax = m.zmax;

driver_config = ClimateMachine.SingleStackConfiguration(
    "BurgersEquation",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

t0 = FT(0)
timeend = FT(10)

Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / max(m.α, m.μh);
Courant_bound = FT(0.1) * Δ
dt = min(Fourier_bound, Courant_bound)

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100 # convert from meters to cm
z_key = "z"
z_label = "z [cm]"
z = get_z(driver_config.grid, z_scale)
state_vars = get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
    i = 1,
    j = 1,
);
aux_vars = get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.dg.state_auxiliary,
    vars_state_auxiliary(m, FT),
    i = 1,
    j = 1,
    exclude = [z_key],
);
all_vars = OrderedDict(state_vars..., aux_vars...);

export_plot_snapshot(
    z,
    all_vars,
    ("ρcT",),
    joinpath(output_dir, "initial_condition_T.png"),
    z_label,
);
export_plot_snapshot(
    z,
    all_vars,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_u.png"),
    z_label,
);
export_plot_snapshot(
    z,
    all_vars,
    ("ρu[2]",),
    joinpath(output_dir, "initial_condition_v.png"),
    z_label,
);

state_vars_var = get_horizontal_variance(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
);

state_vars_avg = get_horizontal_mean(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
);

export_plot_snapshot(
    z,
    state_vars_avg,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_avg_u.png"),
    z_label,
);
export_plot_snapshot(
    z,
    state_vars_var,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_variance_u.png"),
    z_label,
);

const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

dims = OrderedDict(z_key => collect(z));

data_var = Dict([k => Dict() for k in 0:n_outputs]...)
data_var[0] = deepcopy(state_vars_var)

data_avg = Dict([k => Dict() for k in 0:n_outputs]...)
data_avg[0] = deepcopy(state_vars_avg)

step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    solver_config.solver,
) do (init = false)
    state_vars_var = get_horizontal_variance(
        driver_config.grid,
        solver_config.Q,
        vars_state_conservative(m, FT),
    )
    state_vars_avg = get_horizontal_mean(
        driver_config.grid,
        solver_config.Q,
        vars_state_conservative(m, FT),
    )
    step[1] += 1
    data_var[step[1]] = deepcopy(state_vars_var)
    data_avg[step[1]] = deepcopy(state_vars_avg)
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

export_plot(
    z,
    data_avg,
    ("ρu[1]"),
    joinpath(output_dir, "solution_vs_time_u.png"),
    z_label,
    xlabel = "Horizontal mean rho*u",
);
export_plot(
    z,
    data_var,
    ("ρu[1]"),
    joinpath(output_dir, "variance_vs_time_u.png"),
    z_label,
    xlabel = "Horizontal variance rho*u",
);
export_plot(
    z,
    data_avg,
    ("ρcT"),
    joinpath(output_dir, "solution_vs_time_T.png"),
    z_label,
    xlabel = "Horizontal mean rho*c*T",
);
export_plot(
    z,
    data_var,
    ("ρu[3]"),
    joinpath(output_dir, "variance_vs_time_w.png"),
    z_label,
    xlabel = "Horizontal variance rho*w",
);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

