# push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/SphericalTensors/src")

using Test, Random
using SphericalTensors

# push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")
# using DMRG

include("../src/includes.jl")

include("util.jl")

include("auxiliary.jl")
include("states.jl")
include("mpo.jl")
include("mpohamiltonian.jl")

## algorithms
include("algorithms/dmrg.jl")
include("algorithms/tdvp.jl")
include("algorithms/timeevompo.jl")
include("algorithms/schurmpo.jl")

