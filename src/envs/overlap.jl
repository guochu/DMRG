

"""
	struct OverlapCache{_A, _B, _C}

A is the bra, B is ket, ⟨A|B⟩, in iterative algorithms,
A is the output, B is the input
"""
struct OverlapCache{_A, _B, _C} <: AbstractCache
	A::_A
	B::_B
	cstorage::_C
end

Base.length(x::OverlapCache) = length(x.A)

bra(x::OverlapCache) = x.A
ket(x::OverlapCache) = x.B

function environments(psiA::M, psiB::M) where {M <: Union{AbstractMPS, MPO}}
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	hold = r_RR(psiA, psiB)
	L = length(psiA)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[L+1] = hold
	cstorage[1] = l_LL(psiA, psiB)
	for i in L:-1:2
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return OverlapCache(psiA, psiB, cstorage)
end

"""
	leftenv(x::OverlapCache, i::Int) 

environment left to site i
"""
leftenv(x::OverlapCache, i::Int) = x.cstorage[i]
"""
	rightenv(x::OverlapCache, i::Int)

environment right to site i
"""
rightenv(x::OverlapCache, i::Int) = x.cstorage[i+1]