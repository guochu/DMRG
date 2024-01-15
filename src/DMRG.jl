module DMRG

# MPO based finite-size algorithms, including DMRG and TDVP

# verbosity level
# verbosity = 0: absolutely no message
# verbosity = 1: only output important warnings
# verbosity = 2: output important information such as iterative energy...
# verbosity = 3: output verbose information such as MPS truncation..
# verbosity = 4: output verbose information such as the current iterative status..


# auxiliary
export truncdimcutoff, leftdeparallel, rightdeparallel, renyi_entropy, PeriodicArray
export stable_tsvd, stable_tsvd!

# mps
export AbstractMPS, AbstractFiniteMPS, MPSTensor, MPOTensor, MPSBondTensor, SiteOperator, bondtensortype, mpstensortype, mpotensortype
export MPS, canonicalize!,leftorth!, rightorth!, Orthogonalize
export isleftcanonical, isrightcanonical, iscanonical
export bond_dimension, bond_dimensions, distance, space_l, space_r, l_LL, r_RR, isstrict, isoneunit, sector
export ophysical_space, iphysical_space, physical_space, physical_spaces, prodmps, randommps, renyi_entropies
export ExactMPS


# mpo
export AbstractMPO, AbstractFiniteMPO, MPO, PartialMPO, AdjointMPO, prodmpo, randommpo, positions, apply!
export leftdeparallel!, rightdeparallel!, deparallel!, deparallel
export compress!, SVDCompression, Deparallelise, MPOCompression

# mpohamiltonian
export sparsempotensoreltype, AbstractSparseMPOTensor
export SchurMPOTensor, SparseMPOTensor, MPOHamiltonian, SchurMPO, SparseMPO, fromABCD

# environments
export environments, leftenv, rightenv, value

# algorithms
export expectation, expectation_canonical
export DMRG1, DMRG2, DMRG1S, TDVP1, TDVP2, TDVP1S, leftsweep!, rightsweep!, sweep!, compute!, ground_state!, ground_state
export TransferMatrix, ac_prime, ac2_prime
export SubspaceExpansionScheme, OptimalExpansion
export MPSAlgorithm, DMRGAlgorithm, TDVPAlgorithm
export timeevompo, WI, WII, complex_stepper
export ED, exact_diagonalization, exact_timeevolution!, exact_timeevolution


#default settings
module Defaults
	const maxiter = 100 # for DMRG iteration
	const D = 100 # default bond dimension 
	const tolgauge = 1e-14 # for MPS truncation
	const tol = 1e-12 # for DMRG iteration
	const tollanczos = 1.0e-10 # for lanczos eigensolver
	const tolexp = 1.0e-8 # for local eigen in DMRG
	const verbosity = 1
end

abstract type MPSAlgorithm end


using Logging: @warn
using Parameters, Printf

using Reexport
using TupleTools
using KrylovKit
@reexport using SphericalTensors
using SphericalTensors: QR, LQ, SVD, SDD
const TK = SphericalTensors

using LinearAlgebra: LinearAlgebra, Symmetric, eigen, Diagonal

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
include("states/exactmps.jl")
include("states/orth.jl")
include("states/initializers.jl")
include("states/linalg.jl")

# mpo
include("mpo/abstractmpo.jl")
include("mpo/finitempo.jl")
include("mpo/partialmpo.jl")
include("mpo/adjointmpo.jl")
include("mpo/initializers.jl")
include("mpo/linalg.jl")
include("mpo/deparallel.jl")
include("mpo/orth.jl")
include("mpo/compression.jl")

# mpo hamiltonian
include("mpohamiltonian/abstractmpotensor.jl")
include("mpohamiltonian/sparsempotensor.jl")
include("mpohamiltonian/schurmpotensor.jl")
include("mpohamiltonian/mpohamiltonian.jl")
include("mpohamiltonian/transfer.jl")
include("mpohamiltonian/arithmetics.jl")
include("mpohamiltonian/constructor.jl")


# environments
include("envs/environments.jl")

# algorithms
include("algorithms/derivatives.jl")
include("algorithms/transfermatrix.jl")
include("algorithms/expecs.jl")
include("algorithms/expansion/optimalexpand.jl")
include("algorithms/dmrg.jl")
include("algorithms/dmrgexcited.jl")
include("algorithms/tdvp.jl")
include("algorithms/w1w2.jl")
include("algorithms/exactdiag.jl")


end
