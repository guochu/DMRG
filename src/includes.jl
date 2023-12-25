
#default settings
module Defaults
	const maxiter = 100 # for DMRG iteration
	const D = 100
	const tolgauge = 1e-14 # for MPS truncation
	const tol = 1e-12 # for DMRG iteration
	const tollanczos = 1.0e-10
	const tolexp = 1.0e-8 # for local eigen in DMRG
	const verbosity = 1
	# using KrylovKit: GMRES
	# const solver = GMRES(tol=1e-12, maxiter=100)
end

abstract type MPSAlgorithm end


using Logging: @warn
using Parameters, Printf

using KrylovKit
using SphericalTensors
using SphericalTensors: QR, LQ, SVD, SDD
const TK = SphericalTensors

using LinearAlgebra: LinearAlgebra, Symmetric, eigen

# auxiliary
include("auxiliary/periodicarray.jl")
include("auxiliary/truncation.jl")
include("auxiliary/deparallel.jl")
include("auxiliary/simple_lanczos.jl")
include("auxiliary/linalg.jl")

# mps
include("states/abstractmps.jl")
include("states/transfer.jl")
include("states/bondview.jl")
include("states/finitemps.jl")
include("states/scaledmps.jl")
include("states/exactmps.jl")
include("states/orth.jl")
include("states/initializers.jl")
include("states/linalg.jl")

# mpo
include("mpo/abstractmpo.jl")
include("mpo/finitempo.jl")
include("mpo/scaledmpo.jl")
include("mpo/adjointmpo.jl")
include("mpo/initializers.jl")
include("mpo/linalg.jl")
include("mpo/deparallel.jl")
include("mpo/orth.jl")
include("mpo/compression.jl")

# mpo hamiltonian
include("mpohamiltonian/abstractmpotensor.jl")
include("mpohamiltonian/SparseMPOTensor.jl")
include("mpohamiltonian/schurmpotensor.jl")
include("mpohamiltonian/mpohamiltonian.jl")
include("mpohamiltonian/transfer.jl")
include("mpohamiltonian/arithmetics.jl")
include("mpohamiltonian/constructor.jl")


# environments
include("envs/environments.jl")

# algorithms
include("algorithms/derivatives.jl")
include("algorithms/expansion/optimalexpand.jl")
include("algorithms/dmrg.jl")
include("algorithms/dmrgexcited.jl")
include("algorithms/tdvp.jl")
include("algorithms/w1w2.jl")
include("algorithms/exactdiag.jl")
