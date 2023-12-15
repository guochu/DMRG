abstract type MPOCompression end

@with_kw struct SVDCompression <: MPOCompression
	D::Int = Defaults.D
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

@with_kw struct Deparallelise <: MPOCompression
	tol::Float64 = DeparalleliseTol
	verbosity::Int = Defaults.verbosity
end

get_trunc(alg::SVDCompression) = truncdimcutoff(D=alg.D, ϵ=alg.tol, add_back=0)

compress!(h::MPO, alg::SVDCompression) = canonicalize!(h, alg=Orthogonalize(SVD(), get_trunc(alg); normalize=false))
compress!(h::MPO, alg::Deparallelise) = deparallel!(h, tol=alg.tol, verbosity=alg.verbosity)
compress!(h::MPO; alg::MPOCompression = Deparallelise()) = compress!(h, alg)