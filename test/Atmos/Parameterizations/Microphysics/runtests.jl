using Test, MPI

include(joinpath(@__DIR__, "..", "..", "..", "testhelpers.jl"))

@testset "Microphysics tests" begin

    runmpi(joinpath(@__DIR__, "1_unit_tests.jl"))
    runmpi(joinpath(@__DIR__, "2_saturation_adjustment.jl"))
    runmpi(joinpath(@__DIR__, "3_warm_rain.jl"))

end
