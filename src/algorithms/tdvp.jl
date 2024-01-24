abstract type TDVPAlgorithm <: DMRGAlgorithm end

function _exp_evolve(f, dt, x, isherm::Bool, tol::Real)
	tmp, info = exponentiate(f, dt, x, ishermitian=isherm, tol=tol/abs(dt))
	return tmp
end

struct TDVP1{T} <: TDVPAlgorithm
	stepsize::T
	tolexp::Float64
	D::Int 
	ishermitian::Bool
	verbosity::Int
end

TDVP1(; stepsize::Number, D::Int=Defaults.D, tolexp::Float64=Defaults.tolexp, ishermitian::Bool=false, verbosity::Int=1) = TDVP1(
	stepsize, tolexp, D, ishermitian, verbosity)

function _leftsweep!(m::ExpectationCache, alg::TDVP1)
	# increase_bond!(m, D=alg.D)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	# driver = isherm ? Lanczos() : Arnoldi()

	for site in 1:length(mps)-1
		(alg.verbosity > 3) && println("sweeping from left to right at site: $site.")
		# tmp, info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver, tol=alg.tolexp)
		tmp = _exp_evolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], isherm, alg.tolexp)

		mps[site], v = leftorth!(tmp, alg=QR())
		hnew = updateleft(hstorage[site], mps[site], mpo[site], mps[site])

		# v, info = exponentiate(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, driver, tol=alg.tolexp)
		v = _exp_evolve(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, isherm, alg.tolexp)
		mps[site+1] = @tensor tmp[-1 -2; -3] := v[-1, 1] * mps[site+1][1, -2, -3]
		hstorage[site+1] = hnew
	end
	site = length(mps)
	# mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], driver, tol=alg.tolexp)
	mps[site] = _exp_evolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], isherm, alg.tolexp)
end

function _rightsweep!(m::ExpectationCache, alg::TDVP1)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	# driver = isherm ? Lanczos() : Arnoldi()
	
	for site in length(mps)-1:-1:1
		(alg.verbosity > 3) && println("sweeping from right to left at site: $site.")

		v, Q = rightorth(mps[site+1], (1,), (2,3), alg=LQ()) 
		mps[site+1] = permute(Q, (1, 2), (3,))
		hnew = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])

		# v, info = exponentiate(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, driver, tol=alg.tolexp)
		v = _exp_evolve(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, isherm, alg.tolexp)
		hstorage[site+1] = hnew
		mps[site] = mps[site] * v

		# mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver, tol=alg.tolexp)
		mps[site] = _exp_evolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], isherm, alg.tolexp)
	end
end

struct TDVP2{T} <: TDVPAlgorithm
	stepsize::T
	tolexp::Float64
	ishermitian::Bool	
	trunc::TruncationDimCutoff
	verbosity::Int
end
TDVP2(;tolexp::Float64=Defaults.tolexp, ishermitian::Bool=false, verbosity::Int=1, trunc::TruncationDimCutoff=DefaultTruncation, stepsize::Number) = TDVP2(
	stepsize, tolexp, ishermitian, trunc, verbosity)

function Base.getproperty(x::TDVP2, s::Symbol)
	if s == :D
		return x.trunc.D
	elseif s == :ϵ
		return x.trunc.ϵ
	else
		getfield(x, s)
	end
end

function _leftsweep!(m::ExpectationCache, alg::TDVP2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	dt = alg.stepsize
	isherm = alg.ishermitian
	# driver = isherm ? Lanczos() : Arnoldi()

	for site in 1:length(mps)-2
		(alg.verbosity > 3) && println("sweeping from left to right at bond: $site.")
		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		# twositemps, info = exponentiate(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt/2, twositemps, driver, tol=alg.tolexp)
		twositemps = _exp_evolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt/2, twositemps, isherm, alg.tolexp)
		u, s, v, err = stable_tsvd!(twositemps, trunc=trunc)
		mps[site] = u

		sitemps = permute(s * v, (1,2), (3,) )
		hstorage[site+1] = updateleft(hstorage[site], mps[site], mpo[site], mps[site])
		# mps[site+1], info = exponentiate(x->ac_prime(x, mpo[site+1], hstorage[site+1], hstorage[site+2]), -dt/2, sitemps, driver, tol=alg.tolexp)
		mps[site+1] = _exp_evolve(x->ac_prime(x, mpo[site+1], hstorage[site+1], hstorage[site+2]), -dt/2, sitemps, isherm, alg.tolexp)
	end

	site = length(mps)-1
	(alg.verbosity > 3) && println("sweeping from left to right at bond: $site.")
	@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
	# twositemps, info = exponentiate(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt, twositemps, driver, tol=alg.tolexp)
	twositemps = _exp_evolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt, twositemps, isherm, alg.tolexp)
	u, s, v, err = stable_tsvd!(twositemps, trunc=trunc)
	mps[site] = u * s
	mps[site+1] = permute(v, (1,2), (3,))

	hstorage[site+1] = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])
	mps.s[site+1] = s
end


function _rightsweep!(m::ExpectationCache, alg::TDVP2)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	trunc = alg.trunc
	dt = alg.stepsize
	isherm = alg.ishermitian
	# driver = isherm ? Lanczos() : Arnoldi()
	for site in length(mps)-2:-1:1
		(alg.verbosity > 3)  && println("sweeping from right to left at bond: $site.")
		# mps[site+1], info = exponentiate(x->ac_prime(x, mpo[site+1], hstorage[site+1], hstorage[site+2]), -dt/2, mps[site+1], driver, tol=alg.tolexp)
		mps[site+1] = _exp_evolve(x->ac_prime(x, mpo[site+1], hstorage[site+1], hstorage[site+2]), -dt/2, mps[site+1], isherm, alg.tolexp)

		@tensor twositemps[-1 -2; -3 -4] := mps[site][-1, -2, 1] * mps[site+1][1, -3, -4]
		# twositemps, info = exponentiate(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt/2, twositemps, driver, tol=alg.tolexp)
		twositemps = _exp_evolve(x->ac2_prime(x, mpo[site], mpo[site+1], hstorage[site], hstorage[site+2]), dt/2, twositemps, isherm, alg.tolexp)
		u, s, v, err = stable_tsvd!(twositemps, trunc=trunc)
		mps[site] = u * s
		mps[site+1] = permute(v, (1,2), (3,))
		hstorage[site+1] = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])
		mps.s[site+1] = s
	end
end


struct TDVP1S{T, E<:SubspaceExpansionScheme} <: TDVPAlgorithm
	stepsize::T
	tolexp::Float64
	ishermitian::Bool	
	trunc::TruncationDimCutoff
	expan::E
	verbosity::Int
end
TDVP1S(; tolexp::Float64=Defaults.tolexp, ishermitian::Bool=false, verbosity::Int=1, stepsize::Number, 
	trunc::TruncationDimCutoff=DefaultTruncation, expan::SubspaceExpansionScheme=DefaultExpansion) = TDVP1S(
	stepsize, tolexp, ishermitian, trunc, expan, verbosity)

function Base.getproperty(x::TDVP1S, s::Symbol)
	if s == :D
		return x.trunc.D
	elseif s == :ϵ
		return x.trunc.ϵ
	else
		getfield(x, s)
	end
end

function _leftsweep!(m::ExpectationCache, alg::TDVP1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	trunc = alg.trunc
	# driver = isherm ? Lanczos() : Arnoldi()

	for site in 1:length(mps)-1
		(alg.verbosity > 3) && println("sweeping from left to right at site: $site.")
		right_expansion!(m, site, alg.expan, trunc)

		# tmp, info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver, tol=alg.tolexp)
		tmp = _exp_evolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], isherm, alg.tolexp)

		mps[site], v = leftorth!(tmp, alg=QR())
		hnew = updateleft(hstorage[site], mps[site], mpo[site], mps[site])

		# v, info = exponentiate(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, driver, tol=alg.tolexp)
		v = _exp_evolve(x->c_prime(x, hnew, hstorage[site+1]), -dt/2, v, isherm, alg.tolexp)
		mps[site+1] = @tensor tmp[-1 -2; -3] := v[-1, 1] * mps[site+1][1, -2, -3]
		hstorage[site+1] = hnew
	end
	site = length(mps)
	# mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], driver, tol=alg.tolexp)
	mps[site] = _exp_evolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt, mps[site], isherm, alg.tolexp)
end

function _rightsweep!(m::ExpectationCache, alg::TDVP1S)
	mpo = m.mpo
	mps = m.mps
	hstorage = m.env
	dt = alg.stepsize	
	isherm = alg.ishermitian
	trunc = alg.trunc
	# driver = isherm ? Lanczos() : Arnoldi()
	
	for site in length(mps)-1:-1:1
		(alg.verbosity > 3) && println("sweeping from right to left at site: $site.")

		v, Q = rightorth(mps[site+1], (1,), (2,3), alg=LQ()) 
		mps[site+1] = permute(Q, (1, 2), (3,))
		hnew = updateright(hstorage[site+2], mps[site+1], mpo[site+1], mps[site+1])

		# v, info = exponentiate(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, driver, tol=alg.tolexp)
		v = _exp_evolve(x->c_prime(x, hstorage[site+1], hnew), -dt/2, v, isherm, alg.tolexp)
		hstorage[site+1] = hnew
		mps[site] = mps[site] * v

		if site > 1
			left_expansion!(m, site, alg.expan, trunc)
		end

		# mps[site], info = exponentiate(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], driver, tol=alg.tolexp)
		mps[site] = _exp_evolve(x->ac_prime(x, mpo[site], hstorage[site], hstorage[site+1]), dt/2, mps[site], isherm, alg.tolexp)
	end
end



function sweep!(m::ExpectationCache, alg::TDVPAlgorithm; kwargs...)
	_leftsweep!(m, alg; kwargs...)
	_rightsweep!(m, alg; kwargs...)
end



