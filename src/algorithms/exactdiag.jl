struct ED <: MPSAlgorithm 
end

struct ExactCache{H<:Union{MPO, MPOHamiltonian}, M<:ExactMPS, _L, _R} <: AbstractCache
	mpo::H
	state::M
	left::_L
	right::_R
end

function ExactCache(h::Union{MPO, MPOHamiltonian}, state::ExactMPS)
	left, right = init_h_center(h, state)
	return ExactCache(h, state, left, right)
end 

environments(mpo::Union{MPO, MPOHamiltonian}, mps::ExactMPS) = ExactCache(mpo, mps)

function exact_diagonalization(env::ExactCache; num::Int=1, which=:SR, ishermitian::Bool=true)
	driver = ishermitian ? Lanczos() : Arnoldi()
	h = env.mpo
	mps = env.state
	middle_site = mps.center
	left, right = env.left, env.right
	vals,vecs,info = eigsolve(x->ac_prime(x, h[middle_site], left, right), mps[middle_site], num, which, driver)

	(info.converged >= num) || @warn "only $(info.converged) eigenpairs converged"
	states = Vector{typeof(mps)}(undef, num)
	for i in 1:num
		states[i] = copy(mps)
		states[i][middle_site] = vecs[i]
	end
	return vals[1:num], states	
end

function exact_diagonalization(h::Union{MPO, MPOHamiltonian}; num::Int=1, which=:SR, ishermitian::Bool=true, kwargs...)
	physpaces = physical_spaces(h)
	mps = ExactMPS(randn, scalartype(h), physpaces; kwargs...)
	env = ExactCache(h, mps)
	return exact_diagonalization(env; num=num, which=which, ishermitian=ishermitian)
end

function ground_state(h::Union{MPO, MPOHamiltonian}, alg::ED; kwargs...)
	a, b = exact_diagonalization(h; kwargs...)
	return a[1], b[1]
end

function init_h_center(mpo::Union{MPO, MPOHamiltonian}, mps::ExactMPS)
	# (length(mpo) == length(mps)) || throw(DimensionMismatch())
	(spacetype(mpo) == spacetype(mps)) || throw(SpaceMismatch())
	isstrict(mpo) || throw(ArgumentError("operator must be strict"))
	center = mps.center
	right = r_RR(mps, mpo, mps)
	L = length(mps)
	left = l_LL(mps, mpo, mps)
	for i in L:-1:center+1
		right = updateright(right, mps[i], mpo[i], mps[i])
	end
	for i in 1:center-1
		left = updateleft(left, mps[i], mpo[i], mps[i])
	end
	return left, right
end

# function exact_diagonalization(h::Union{MPO, MPOHamiltonian}; ishermitian::Bool, num::Int=1, which=:SR, kwargs...) 
# 	len = length(h)
# 	driver = ishermitian ? Lanczos() : Arnoldi()

# 	physpaces = physical_spaces(h)
# 	if isa(h, MPOHamiltonian)
# 		physpaces = [physpaces[i] for i in 1:len]
# 	end

# 	mps = ExactMPS(randn, scalartype(h), physpaces; kwargs...)
# 	middle_site = mps.center

# 	left, right = init_h_center(h, mps)

# 	vals,vecs,info = eigsolve(x->ac_prime(x, h[middle_site], left, right), mps[middle_site], num, which, driver)

# 	(info.converged >= num) || @warn "only $(info.converged) converged."
# 	states = Vector{typeof(mps)}(undef, num)
# 	for i in 1:num
# 		states[i] = copy(mps)
# 		states[i][middle_site] = vecs[i]
# 	end
# 	return vals[1:num], states
# end

function exact_timeevolution!(env::ExactCache, t::Number; ishermitian::Bool=true)
	driver = ishermitian ? Lanczos() : Arnoldi()
	h = env.mpo
	psi = env.state
	middle_site = psi.center
	left, right = env.left, env.right
	mpsj, info = exponentiate(x->ac_prime(x, h[middle_site], left, right), t, psi[middle_site], driver)
	(info.converged >= 1) || error("fail to converge")
	psi[middle_site] = mpsj
	return env
end

function exact_timeevolution(h::Union{MPO, MPOHamiltonian}, t::Number, psi::ExactMPS; ishermitian::Bool)
	env = ExactCache(h, copy(psi))
	env = exact_timeevolution!(env, t, ishermitian=ishermitian)
	return env.state
end

