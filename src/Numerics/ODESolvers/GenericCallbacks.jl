"""
A set of callback functions to be used with an `AbstractODESolver`
"""
module GenericCallbacks
using MPI

"""
    EveryXWallTimeSeconds(f, time, mpicomm)

This callback will run the function 'f()' every `time` wallclock time seconds.
The `mpicomm` is used to syncronize runtime across MPI ranks.
"""
mutable struct EveryXWallTimeSeconds
    "time of the last callback"
    lastcbtime_ns::UInt64
    "time between callbacks"
    Δtime::Real
    "MPI communicator"
    mpicomm::MPI.Comm
    "function to execute for callback"
    func
    "function to call to initialize the callback"
    init
    function EveryXWallTimeSeconds(func, Δtime, mpicomm, init = () -> nothing)
        lastcbtime_ns = zero(UInt64)
        new(lastcbtime_ns, Δtime, mpicomm, func, init)
    end
end

function initialize!(cb::EveryXWallTimeSeconds, solver, Q, param, t0)
    cb.lastcbtime_ns = time_ns()
    cb.init()
end

function (cb::EveryXWallTimeSeconds)(solver, Q, param, t)
    # Check whether we should do a callback
    currtime_ns = time_ns()
    runtime = (currtime_ns - cb.lastcbtime_ns) * 1e-9
    runtime = MPI.Allreduce(runtime, MPI.MAX, cb.mpicomm)
    if runtime < cb.Δtime
        return 0
    else
        # Compute the next time to do a callback
        cb.lastcbtime_ns = currtime_ns
        return cb.func()
    end
end




"""
   EveryXSimulationTime(f, time, state)

This callback will run the function 'f()' every `time` wallclock time seconds.
The `state` is used to query for the simulation time.
"""
mutable struct EveryXSimulationTime
    "time of the last callback"
    lastcbtime::Real
    "time between callbacks"
    Δtime::Real
    "function to execute for callback"
    func
    "function to call to initialize the callback"    
    init
    function EveryXSimulationTime(func, Δtime, init=() -> nothing)
        lastcbtime = 0
        new(0, Δtime, func, init)
    end
end

function initialize!(cb::EveryXSimulationTime, solver, Q, param, t0)
    cb.lastcbtime = t0
    cb.init()
end
function (cb::EveryXSimulationTime)(solver, Q, param, t)
    # Check whether we should do a callback
    if (t - cb.lastcbtime) < cb.Δtime
        return 0
    else
        # Compute the next time to do a callback
        cb.lastcbtime = t
        return cb.func()
    end
end


"""
   EveryXSimulationSteps(f, steps[, init = ()->nothing])

This callback will run the function 'f()' every `steps` of the time stepper
"""
struct EveryXSimulationSteps
    "number of steps since last callback"
    steps::Int
    "number of steps between callbacks"
    Δsteps::Int
    "function to execute for callback"
    func
    "function to call to initialize the callback"    
    init
    function EveryXSimulationSteps(func, Δsteps, init = () -> nothing)
        new(0, Δsteps, func, init)
    end
end
function initialize!(cb::EveryXSimulationTime, solver, Q, param, t0)
    cb.steps = 0
    cb.init()
end

function (cb::EveryXSimulationSteps)(solver, Q, param, t)
    cb.steps += 1
    if cb.steps < cb.Δsteps
        return 0
    else
        cb.steps = 0
        return cb.func()
    end
end

end
