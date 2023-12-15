"""
	struct SparseMPOTensor{S<:ElementarySpace, M<:MPOTensor, T<:Number} <: AbstractSparseMPOTensor{S}

leftspaces are left space, rightspaces are right space
"""
struct SparseMPOTensor{S<:ElementarySpace, M<:MPOTensor, T<:Number} <: AbstractSparseMPOTensor{S}
	Os::Array{Union{M, T}, 2}
	leftspaces::Vector{S}
	rightspaces::Vector{S}
	pspace::S
end
Base.copy(x::SparseMPOTensor) = SparseMPOTensor(copy(x.Os), copy(x.leftspaces), copy(x.rightspaces), x.pspace)
TK.scalartype(::Type{SparseMPOTensor{S,M,T}}) where {S,M,T} = T

function SparseMPOTensor{S, M, T}(data::AbstractMatrix{Any}) where {S<:ElementarySpace, M <:MPOTensor, T<:Number}
	Os, leftspaces, rightspaces, pspace = compute_mpotensor_data(S, M, T, data)
	return SparseMPOTensor{S, M, T}(Os, leftspaces, rightspaces, pspace)
end

function SparseMPOTensor(data::AbstractMatrix{Any}, leftspaces::Vector{S}, rightspaces::Vector{S}, pspace::S) where {S <: ElementarySpace}
	T = compute_scalartype(data)
	M = tensormaptype(S, 2, 2, T)
	Os = compute_mpotensor_data(M, T, data, leftspaces, rightspaces, pspace)
	return SparseMPOTensor{S, M, T}(Os, leftspaces, rightspaces, pspace)
end

"""
	SparseMPOTensor(data::Array{Any, 2}) 
"""
function SparseMPOTensor(data::AbstractMatrix{Any}) 
	S, T = compute_generic_mpotensor_types(data)
	M = tensormaptype(S, 2, 2, T)
	return SparseMPOTensor{S, M, T}(data)
end

Base.contains(m::SparseMPOTensor{S, M, T}, i::Int, j::Int) where {S, M, T} = (m.Os[i, j] != zero(T))
function isscal(x::SparseMPOTensor{S,M,T}, i::Int, j::Int) where {S,M,T}
	sj = x.Os[i, j]
	return (sj isa T) && (abs(sj) > 1.0e-14)
end 


function Base.setindex!(m::SparseMPOTensor{S, M, T}, v, i::Int, j::Int) where {S, M, T}
	if isa(v, Number)
		m.Os[i, j] = convert(T, v)
	elseif isa(v, MPOTensor)
		((space(v, 1) == m.leftspaces[i]) && (space(v, 3)' == m.rightspaces[j])) || throw(SpaceMismatch())
		m.Os[i, j] = convert(M, v) 
	elseif isa(v, MPSBondTensor)
		b_iden = isomorphism(Matrix{T}, m.leftspaces[i], m.rightspaces[j])
		@tensor tmp[-1 -2; -3 -4] := b_iden[-1, -3] * v[-2, -4]
		m.Os[i, j] = tmp
	else
		throw(ArgumentError("input should be scalar, MPOTensor or MPSBondTensor type"))
	end
end