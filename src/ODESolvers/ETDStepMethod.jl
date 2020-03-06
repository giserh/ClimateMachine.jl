module ETDStepMethod
using ..ODESolvers
using ..MPIStateArrays: device, realview

using StaticArrays

using GPUifyLoops
#include("ETDStepMethod_kernels.jl")


export EB4, EB1

ODEs = ODESolvers

"""
    TimeScaledRHS(a, b, rhs!)

When evaluated at time `t`, evaluates `rhs!` at time `a + bt`.
"""
mutable struct TimeScaledRHS{RT}
  a::RT
  b::RT
  rhs!
end

function (o::TimeScaledRHS)(dQ, Q, params, tau; increment)
  o.rhs!(dQ, Q, params, o.a + o.b*tau; increment=increment)
end

using GPUifyLoops

"""
ETDStep(slowrhs!, fastrhs!, fastmethod,
                           α, β, γ,
                           Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

This is a time stepping object for explicitly time stepping the partitioned differential
equation given by right-hand-side functions `f_fast` and `f_slow` with the state `Q`, i.e.,

```math
  \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds an exponential time differencing  Runge-Kutta scheme
based on the provided `β` tableaux and `fastmethod` for solving
the fast modes.

### References
    @article{Krogstad2005,
      title={Generalized integrating factor methods for stiff PDEs},
      author={Krogstad, Stein},
      journal= {Journal of Computational Physics},
      volume= {203},
      number= {1},
      pages = {72 - 88},
      year = {2005},
    }
"""
mutable struct ETDStep{T, RT, AT, FS, Nstages, Nstagesm1, Nstagesm2, Nstages_sq} <: ODEs.AbstractODESolver
  "time step"
  dt::RT
  "time"
  t::RT
  #"storage for y_n"
  #yn::AT
  "Storage for ``f(Y_nj)``"
  fYnj::NTuple{Nstagesm1, AT}
  "Storage for offset"
  offset::AT
  "slow rhs function"
  slowrhs!
  "RHS for fast solver"
  tsfastrhs!::TimeScaledRHS{RT}
  "fast rhs method"
  fastsolver::FS
#  "number of substeps per stage"
#  nsubsteps::Int

  nStages::Int64
  nPhi::Int64
  nPhiStages::Array{Int64,1}

# α::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  #β::SArray{NTuple{2, Nstages}, AT, 2, Nstages_sq}
  #βS::SArray{NTuple{2, Nstages}, AT, 2, Nstages_sq}
  β::Array{Array{Float64,1},2};
  βS::Array{Array{Float64,1},2};

# γ::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
  d::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
  c::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
# c̃::SArray{NTuple{1, Nstages}, RT, 1, Nstages}

  function ETDStep(slowrhs!, fastrhs!, fastmethod,
                                      Nstages, nPhi, nPhiStages, β, βS, c,
                                      Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

    T = eltype(Q)
    RT = real(T)


    #yn = similar(Q)
#   ΔYnj = ntuple(_ -> similar(Q), Nstages-2)
    fYnj = ntuple(_ -> similar(Q), Nstages-1)
    offset = similar(Q)
    tsfastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)
    fastsolver = fastmethod(tsfastrhs!, Q)

    #d = sum(β, dims=2)

    #c = similar(d)
    d=copy(c);

#   for i = eachindex(c)

#     c[i] = d[i]
#     if i > 1
#       c[i] += sum(j-> (α[i,j] + γ[i,j])*c[j], 1:i-1)
#     end
#   end
#   c̃ = α*c

#   new{T, RT, AT, typeof(fastsolver), Nstages, Nstages-1, Nstages-2, Nstages ^ 2}(RT(dt), RT(t0), yn, ΔYnj, fYnj, offset,
#                                        slowrhs!, tsfastrhs!, fastsolver,
#                                        α, β, γ, d, c, c̃)
    new{T, RT, AT, typeof(fastsolver), Nstages, Nstages-1, Nstages-2, Nstages ^ 2}(RT(dt), RT(t0), fYnj, offset,
                                         slowrhs!, tsfastrhs!, fastsolver,
                                         Nstages, nPhi, nPhiStages, β, βS, d, c)
  end
end


# TODO almost identical functions seem to be defined for every ode solver,
# define a common one in ODEsolvers ?
function ODEs.dostep!(Q, etd::ETDStep, param,
                      timeend::AbstractFloat, adjustfinalstep::Bool)
  time, dt = etd.t, etd.dt
  @assert dt > 0
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  ODEs.dostep!(Q, etd, param, time, dt)

  if dt == etd.dt
    etd.t += dt
  else
    etd.t = timeend
  end
  return etd.t
end

function ODEs.dostep!(Q, etd::ETDStep, p,
                      time::AbstractFloat, dt::AbstractFloat)
  FT = eltype(dt)
  β = etd.β
  βS = etd.βS
  nPhi = etd.nPhi
  #yn = etd.yn
  fYnj = etd.fYnj
  offset = etd.offset
  d = etd.d
  c = etd.c
  #c̃ = etd.c̃
  slowrhs! = etd.slowrhs!
  fastsolver = etd.fastsolver
  fastrhs! = etd.tsfastrhs!

  nStages = etd.nStages

  #copyto!(yn, Q) # first stage
  for iStage = 1:nStages-1
    slowrhs!(fYnj[iStage], Q, p, time + c[iStage]*dt, increment=false)

    nsteps = cld(dt, fastsolver.max_inner_dt)
    inner_dt = dt / nsteps

    nsLoc=Int64(ceil(nsteps*c[iStage+1]));
    dtLoc=Float64(dt)*c[iStage+1];
    dτ=dtLoc/nsLoc;

    # TODO: we want to be able to write
    #   solve!(Q, fastsolver, p; numberofsteps = mis.nsubsteps)  #(1c)
    # especially if we want to use StormerVerlet, but need some way to pass in `offset`
    ODEs.dostep!(Q, fastsolver, p, nsLoc, time, dτ, iStage, β, βS, nPhi, fYnj, FT(1), realview(offset), nothing)  #(1c)
  end
end

function EB1(slowrhs!, fastrhs!, fastmethod, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

  nStages=2;
  nPhi=1;

  nPhiStages=[0, 1]; #???

  β = [[[0.0]] [[0.0]];
      [[1.0]] [[0.0]]];
  βS =[[[0.0]] [[0.0]];
      [[0.0]] [[0.0]]];


  c = [0.0, 1.0, 1.0];

  for i=2:2
    for j=1:i-1
    kFac=1;
      for k=1:1
        kFac=kFac*max(k-1,1)*c[i];
        βS[i,j][k]=β[i,j][k]/kFac;
      end
    end
  end

  ETDStep(slowrhs!, fastrhs!, fastmethod, nStages, nPhi, nPhiStages, β, βS, c, Q; dt=dt, t0=t0)
end


function EB4(slowrhs!, fastrhs!, fastmethod, Q::AT; dt=0, t0=0) where {AT <: AbstractArray}

  nStages=5;
  nPhi=3;

  nPhiStages=[0,1,2,2,3];

  β = [[[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[0.5,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[0.5,-1.0,0.0]] [[0.0,1.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[1.0,-2.0,0.0]] [[0.0,0.0,0.0]]  [[0.0,2.0,0.0]]  [[0.0,0.0,0.0]]  [[0.0,0.0,0.0]];
       [[1.0,-3.0,4.0]] [[0.0,2.0,-4.0]] [[0.0,2.0,-4.0]] [[0.0,-1.0,4.0]] [[0.0,0.0,0.0]]];

  #βS=similar(β);
  βS = [[[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]];
        [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]] [[0.0,0.0,0.0]]];

  c = [0.0, 0.5, 0.5, 1.0, 1.0];
     #c[i] is usually sum of first elements in i-th row)

  for i=2:nStages
    for j=1:i-1
      kFac=1;
        for k=1:nPhi
          kFac=kFac*max(k-1,1);
          βS[i,j][k]=β[i,j][k]/(kFac*c[i]);
          β[i,j][k]=β[i,j][k]/c[i];
        end
    end
  end

  #=γ = [0  0               0              0;
       0  0               0              0;
       0  0.652465126004  0              0;
       0 -0.0732769849457 0.144902430420 0]=# #not needed (yet?)

  ETDStep(slowrhs!, fastrhs!, fastmethod, nStages, nPhi, nPhiStages, β, βS, c, Q; dt=dt, t0=t0)
end

end