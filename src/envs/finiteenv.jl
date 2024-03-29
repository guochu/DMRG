function init_hstorage(mpo::Union{MPO, MPOHamiltonian}, mps::MPS, center::Int)
	(length(mpo) == length(mps)) || throw(DimensionMismatch())
	(spacetype(mpo) == spacetype(mps)) || throw(SpaceMismatch())
	isstrict(mpo) || throw(ArgumentError("operator must be strict"))
	right = r_RR(mps, mpo, mps)
	L = length(mps)
	hstorage = Vector{typeof(right)}(undef, L+1)
	hstorage[L+1] = right
	hstorage[1] = l_LL(mps, mpo, mps)
	for i in L:-1:center+1
		hstorage[i] = updateright(hstorage[i+1], mps[i], mpo[i], mps[i])
	end
	for i in 1:center-1
		hstorage[i+1] = updateleft(hstorage[i], mps[i], mpo[i], mps[i])
	end
	return hstorage
end

init_hstorage_right(mpo::Union{MPO, MPOHamiltonian}, mps::MPS) = init_hstorage(mpo, mps, 1)


struct ExpectationCache{M<:Union{MPO, MPOHamiltonian}, V<:MPS, H} <: AbstractCache
	mpo::M
	mps::V
	hstorage::H
end

environments(mpo::Union{MPO, MPOHamiltonian}, mps::MPS) = ExpectationCache(mpo, mps, init_hstorage_right(mpo, mps))

function Base.getproperty(m::ExpectationCache, s::Symbol)
	if s == :state
		return m.mps
	elseif s == :h
		return m.mpo
	elseif s == :env 
		return m.hstorage
	else
		return getfield(m, s)
	end
end


function updateleft!(env::ExpectationCache, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
end

function updateright!(env::ExpectationCache, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
end
leftenv(x::ExpectationCache, i::Int) = x.hstorage[i]
rightenv(x::ExpectationCache, i::Int) = x.hstorage[i+1]


# for excited states
struct ProjectedExpectationCache{M<:Union{MPO, MPOHamiltonian}, V<:MPS, H, C} <: AbstractCache
	mpo::M
	mps::V
	projectors::Vector{V}
	hstorage::H
	cstorages::Vector{C}
end

function init_cstorage_right(psiA::MPS, psiB::MPS)
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	L = length(psiA)
	hold = r_RR(psiA, psiB)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[1] = l_LL(psiA)
	cstorage[L+1] = hold
	for i in L:-1:2
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return cstorage
end


environments(mpo::Union{MPO, MPOHamiltonian}, mps::M, projectors::Vector{M}) where {M <: MPS} = ProjectedExpectationCache(
	mpo, mps, projectors, init_hstorage_right(mpo, mps), [init_cstorage_right(mps, item) for item in projectors])

function Base.getproperty(m::ProjectedExpectationCache, s::Symbol)
	if s == :state
		return m.mps
	elseif s == :h
		return m.mpo
	elseif s == :env 
		return m.hstorage
	elseif s == :cenvs
		return m.cstorages
	else
		return getfield(m, s)
	end
end


function updateleft!(env::ProjectedExpectationCache, site::Int)
	env.hstorage[site+1] = updateleft(env.hstorage[site], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site+1] = updateleft(env.cstorages[l][site], env.mps[site], env.projectors[l][site])
	end
end


function updateright!(env::ProjectedExpectationCache, site::Int)
	env.hstorage[site] = updateright(env.hstorage[site+1], env.mps[site], env.h[site], env.mps[site])
	for l in 1:length(env.cstorages)
	    env.cstorages[l][site] = updateright(env.cstorages[l][site+1], env.mps[site], env.projectors[l][site])
	end
end

