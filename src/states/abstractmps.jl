const MPSTensor{S} = AbstractTensorMap{S, 2, 1} where {S<:ElementarySpace}
const MPOTensor{S} = AbstractTensorMap{S, 2, 2} where {S<:ElementarySpace}
# The bond tensor are just the singlur vector but has to be stored as a general matrix 
# since TensorKit does not specialize for Diagonal Matrices
const MPSBondTensor{S} = AbstractTensorMap{S, 1, 1} where {S<:ElementarySpace}
const SiteOperator{S} = Union{MPOTensor{S}, MPSBondTensor{S}}

# bondtensortype(::Type{A}) where {A <: AbstractTensorMap} = tensormaptype(spacetype(A), 1, 1, TK.similarstoragetype(A, real(scalartype(A))))
# mpstensortype(::Type{A}) where {A<:AbstractTensorMap} = tensormaptype(spacetype(A), 2, 1, storagetype(A))
# mpotensortype(::Type{A}) where {A<:AbstractTensorMap} = tensormaptype(spacetype(A), 2, 2, storagetype(A))

mpstensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T} = tensormaptype(S, 2, 1, T)
mpotensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T} = tensormaptype(S, 2, 2, T)
bondtensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T<:Union{Number, DenseMatrix}} = tensormaptype(S, 1, 1, T)
bondtensortype(::Type{S}, ::Type{T}) where {S <: ElementarySpace, T<:Diagonal} = diagonalmaptype(S, eltype(T))

# bondtensortype(a::AbstractTensorMap) = bondtensortype(typeof(a))
# mpstensortype(a::AbstractTensorMap) = mpstensortype(typeof(a))
# mpotensortype(a::AbstractTensorMap) = mpotensortype(typeof(a))



abstract type AbstractMPS{A<:MPSTensor} end

TK.scalartype(::Type{<:AbstractMPS{A}}) where {A<:MPSTensor} = scalartype(A)
TK.spacetype(::Type{<:AbstractMPS{A}}) where {A<:MPSTensor} = spacetype(A)
TK.spacetype(m::AbstractMPS) = spacetype(typeof(m))
TK.sectortype(A::Type{<:AbstractMPS}) = sectortype(spacetype(A))
TK.sectortype(a::AbstractMPS) = sectortype(typeof(a))
mpstensortype(::Type{<:AbstractMPS{A}}) where {A<:MPSTensor} = A
mpstensortype(m::AbstractMPS) = mpstensortype(typeof(m))

storage(a::AbstractMPS) = error("storage not implemented for mpstype $(typeof(a))")
Base.length(a::AbstractMPS) = length(storage(a))
Base.isempty(a::AbstractMPS) = isempty(storage(a))
Base.getindex(a::AbstractMPS, i::Int) = getindex(storage(a), i)
Base.firstindex(a::AbstractMPS) = firstindex(storage(a))
Base.lastindex(a::AbstractMPS) = lastindex(storage(a))

# conventions used
"""
	space_l(psi::AbstractMPS)

Return the left auxiliary space of MPS or MPO, by convention it has to be vacuum
The returned space is a dual space
"""
space_l(a::AbstractMPS) = space_l(a[1])
"""
	space_r(psi::AbstractMPS) 

Return the right auxiliary space of MPS or MPO, it can be nontrivial
The returned space is a normal space
"""
space_r(a::AbstractMPS) = space_r(a[end])
"""
	sector(a::AbstractMPS)

Return the sector of MPS, this function assumes that MPS
has a single sector
"""
sector(a::AbstractMPS) = _sector(a)
function _sector(a)
	s = sectors(space_r(a)')
	(length(s) == 1) || throw(ArgumentError("multiple sectors not allowed"))
	return first(s)
end
space_l(a::MPSBondTensor) = space(a, 1)
space_r(a::MPSBondTensor) = space(a, 2)
space_l(a::MPSTensor) = space(a, 1)
space_r(a::MPSTensor) = space(a, 3)
space_l(a::MPOTensor) = space(a, 1)
space_r(a::MPOTensor) = space(a, 3)
physical_space(a::MPSTensor) = space(a, 2)
ophysical_space(a::MPOTensor) = space(a, 2)
iphysical_space(a::MPOTensor) = space(a, 4)
ophysical_space(a::MPSBondTensor) = space(a, 1)
iphysical_space(a::MPSBondTensor) = space(a, 2)
physical_space(a::SiteOperator) = ophysical_space(a)

"""
	r_RR(a::MPS, b::MPS)

Notice the convention!!!
a is bra, b is ket, ^
					a
					-
					b
					v
for r_RR b is codomain, namely 
		----2
r_RR =
		----1
for l_LL a is codomain, namely
		----1
l_LL =
		----2
"""
r_RR(a::MPSTensor, b::MPSTensor) = loose_isometry(Matrix{promote_type(scalartype(a), scalartype(b))}, space_r(b)', space_r(a)')
l_LL(a::MPSTensor, b::MPSTensor) = loose_isometry(Matrix{promote_type(scalartype(a), scalartype(b))}, space_l(a), space_l(b))
r_RR(a::AbstractMPS, b::AbstractMPS) = loose_isometry(Matrix{promote_type(scalartype(a), scalartype(b))}, space_r(b)', space_r(a)')
l_LL(a::AbstractMPS, b::AbstractMPS) = loose_isometry(Matrix{promote_type(scalartype(a), scalartype(b))}, space_l(a), space_l(b))


"""
	r_RR, right boundary 2-tensor
	i-1
	o-2
"""
r_RR(a::AbstractMPS) = r_RR(a, a)
"""
	l_LL, left boundary 2-tensor
	i-1
	o-2
"""
l_LL(a::AbstractMPS) = l_LL(a, a)

isoneunit(s::ElementarySpace) = isdual(s) ? dual(s) == oneunit(s) : s == oneunit(s) 
"""
	isstrict(a::AbstractMPS) 

isstrict means that right space is vacuum in case of non-Abelian symmetry
For MPS we generally allow arbitrary sector 
for MPO many functions only allow strict MPO, especially those operates
on both MPO and AdjointMPO
"""
isstrict(a::AbstractMPS) = _isstrict(a)
function _isstrict(a)
	@assert isoneunit(space_l(a))
	return isstrict(space_r(a))
end 


"""
	bond_dimension(psi::MPS[, bond::Int])
	bond_dimension(h::MPO[, bond::Int])

Return bond dimension at the given bond, or return the largest bond dimension of all bonds.
"""
bond_dimension(a::AbstractMPS, bond::Int) = begin
	((bond >= 1) && (bond <= length(a))) || throw(BoundsError(storage(a), bond))
	dim(space(a[bond], 3))
end 
bond_dimensions(a::AbstractMPS) = [bond_dimension(a, i) for i in 1:length(a)]
bond_dimension(a::AbstractMPS) = maximum(bond_dimensions(a))

physical_space(a::AbstractMPS, i::Int) = physical_space(a[i])
"""
	physical_spaces(psi::MPS)
	physical_spaces(psi::MPO) 
Return all the physical spaces of MPS or MPO
"""
physical_spaces(a::AbstractMPS) = [physical_space(a[i]) for i in 1:length(a)]

# check if mps tensor is canonical
function isleftcanonical(psij::MPSTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[1,2,-1]) * psij[1,2,-2]
	return isapprox(r, one(r); kwargs...) 
end
function isleftcanonical_r(psij::MPSTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[1,2,-1]) * psij[1,2,-2]
	for (c, b) in blocks(r)
		_one = lmul!( 1/(dim(c) * size(b, 1)), one(b))
		isapprox(b, _one; kwargs...) || return false
	end
	return true
end
function isrightcanonical(psij::MPSTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[-1,1,2]) * psij[-2,1,2]
	return isapprox(r, one(r); kwargs...) 
end

# check if mpo tensor is canonical
function isleftcanonical(psij::MPOTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[1,2,-1,3]) * psij[1,2,-2,3]
	return isapprox(r, one(r); kwargs...) 
end
function isleftcanonical_r(psij::MPOTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[1,2,-1,3]) * psij[1,2,-2,3]
	for (c, b) in blocks(r)
		_one = lmul!( 1/(dim(c) * size(b, 1)), one(b))
		isapprox(b, _one; kwargs...) || return false
	end
	return true
end
function isrightcanonical(psij::MPOTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[-1,1,2,3]) * psij[-2,1,2,3]
	return isapprox(r, one(r); kwargs...) 
end

isstrict(s::ElementarySpace) = isoneunit(s) || (FusionStyle(sectortype(s)) isa UniqueFusion)

# check_mpstensor_dir(m::MPSTensor) = (!isdual(space(m, 1))) && (!isdual(space(m, 2))) && isdual(space(m, 3))
# check_bondtensor_dir(m::MPSBondTensor) = (!isdual(space(m, 1))) && isdual(space(m, 2))
# check_mpotensor_dir(m::MPOTensor) = (!isdual(space(m, 1))) && (!isdual(space(m, 2))) && isdual(space(m, 3)) && isdual(space(m, 4))

common_scalartype(a, b) = promote_type(scalartype(a), scalartype(b))
common_scalartype(a, b, c) = promote_type(scalartype(a), scalartype(b), scalartype(c))
common_storagetype(a, b) = Matrix{common_scalartype(a, b)}
common_storagetype(a, b, c) = Matrix{common_scalartype(a, b, c)}