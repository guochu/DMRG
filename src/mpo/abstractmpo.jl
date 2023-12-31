abstract type AbstractMPO{A<:MPOTensor} end

TK.scalartype(::Type{<:AbstractMPO{A}}) where {A<:MPOTensor} = scalartype(A)
TK.spacetype(::Type{<:AbstractMPO{A}}) where {A<:MPOTensor} = spacetype(A)
TK.spacetype(m::AbstractMPO) = spacetype(typeof(m))

storage(a::AbstractMPO) = error("storage not implemented for mpotype $(typeof(a))")
Base.length(a::AbstractMPO) = length(storage(a))
Base.isempty(a::AbstractMPO) = isempty(storage(a))
Base.getindex(a::AbstractMPO, i::Int) = getindex(storage(a), i)
Base.firstindex(a::AbstractMPO) = firstindex(storage(a))
Base.lastindex(a::AbstractMPO) = lastindex(storage(a))

space_l(state::AbstractMPO) = space_l(state[1])
space_r(state::AbstractMPO) = space_r(state[end])
sector(a::AbstractMPO) = _sector(a)


r_RR(a::AbstractMPO, b::AbstractMPO) = loose_isometry(common_storagetype(psiA, psiB), space_r(b)', space_r(a)')
l_LL(psiA::AbstractMPO, psiB::AbstractMPO) = loose_isometry(common_storagetype(psiA, psiB), space_l(psiA), space_l(psiB))


"""
	r_RR, right boundary 2-tensor
	i-1
	o-2
"""
r_RR(a::AbstractMPO) = r_RR(a, a)
"""
	l_LL, left boundary 2-tensor
	o-1
	i-2
"""
l_LL(a::AbstractMPO) = l_LL(a, a)

isstrict(h::AbstractMPO) = _isstrict(h)


ophysical_space(a::AbstractMPO, i::Int) = ophysical_space(a[i])
iphysical_space(a::AbstractMPO, i::Int) = iphysical_space(a[i])
function physical_spaces(psi::AbstractMPO)
	xs = ophysical_spaces(psi)
	(xs == adjoint.(iphysical_spaces(psi))) || throw(SpaceMismatch("i and o physical dimension mismatch."))
	return xs
end

ophysical_spaces(psi::AbstractMPO) = [ophysical_space(psi[i]) for i in 1:length(psi)]
iphysical_spaces(psi::AbstractMPO) = [iphysical_space(psi[i]) for i in 1:length(psi)]

bond_dimensions(h::AbstractMPO) = [bond_dimension(h, i) for i in 1:length(h)]
bond_dimension(h::AbstractMPO) = maximum(bond_dimensions(h))