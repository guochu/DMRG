# push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/SphericalTensors/src")

using Test, Random
using SphericalTensors

# push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")
# using DMRG

include("../src/includes.jl")

include("util.jl")

Random.seed!(3247)

include("auxiliary.jl")
include("states.jl")
include("mpo.jl")
include("mpohamiltonian.jl")

## algorithms
include("algorithm/dmrg.jl")
include("algorithm/tdvp.jl")
include("algorithm/timeevompo.jl")

