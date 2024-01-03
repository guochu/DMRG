"""
	struct MPOHamiltonian{M <: AbstractSparseMPOTensor}

A generic MPO which stores a chain of AbstractSparseMPOTensor (Matrix of MPOTensors)

For finite system, the first site tensor is understood as the first row of the 
first AbstractSparseMPOTensor, and the last site tensor is understood as the last 
column of the last AbstractSparseMPOTensor
"""
struct MPOHamiltonian{M <: AbstractSparseMPOTensor}
	data::Vector{M}

function MPOHamiltonian{M}(data::AbstractVector) where {M <: AbstractSparseMPOTensor}
	@assert !isempty(data) 
	(data[1].leftspaces == data[end].rightspaces) || throw(SpaceMismatch())
	for i in 1:length(data)-1
		(size(data[i], 2) == size(data[i+1], 1)) || throw(DimensionMismatch())

	end
	new{M}(convert(Vector{M}, data))
end

end

Base.length(x::MPOHamiltonian) = length(x.data)
Base.getindex(x::MPOHamiltonian, i::Int) = getindex(x.data, i)
Base.firstindex(x::MPOHamiltonian) = firstindex(x.data)
Base.lastindex(x::MPOHamiltonian) = lastindex(x.data)
Base.copy(x::MPOHamiltonian) = MPOHamiltonian(copy(x.data))

TK.spacetype(::Type{MPOHamiltonian{M}}) where M = spacetype(M)
TK.spacetype(x::MPOHamiltonian) = spacetype(typeof(x))
TK.scalartype(::Type{MPOHamiltonian{M}}) where {M} = scalartype(M)

MPOHamiltonian(data::AbstractVector{M}) where {M <: AbstractSparseMPOTensor} = MPOHamiltonian{M}(data)
function MPOHamiltonian(data::Vector{Matrix{Any}}, virtualspaces::Vector{Vector{S}}, pspaces::Vector{S}) where {S<:ElementarySpace}
	@assert length(data) == length(pspaces) == length(virtualspaces)-1
	return MPOHamiltonian([SparseMPOTensor(data[i], virtualspaces[i], virtualspaces[i+1], pspaces[i]) for i in 1:length(data)])
end
MPOHamiltonian(data::Vector{Matrix{Any}}) = MPOHamiltonian([SparseMPOTensor(item) for item in data])

SparseMPO(data::AbstractVector{<:SparseMPOTensor}) = MPOHamiltonian(data)
SchurMPO(data::AbstractVector{<:SchurMPOTensor}) = MPOHamiltonian(data)

function Base.getindex(x::MPOHamiltonian, i::Int, j::Int, k::Int)
	x[i][j, k]
end 

Base.getindex(x::MPOHamiltonian, i::Colon, j::Int, k::Int) = [getindex(x, i,j,k) for i in 1:length(x)]

# isscal(x::MPOHamiltonian, a::Int, b::Int, c::Int) = isscal(x[a], b, c)

# """
# 	isid(ham::MPOHamiltonian,i::Int)

# Check if ham[:,i,i] = 1 for every i
# """
# function isid(ham::MPOHamiltonian,i::Int)
# 	r = true
# 	for b in 1:length(ham)
# 		r = r && isscal(ham[b],i,i) && abs(ham[b].Os[i,i]-one(eltype(ham)))<1e-14
# 	end
# 	return r
# end 


space_l(x::MPOHamiltonian) = x[1].leftspaces[1]
space_r(x::MPOHamiltonian) = (x[end].rightspaces[end])'

bond_dimension(h::MPOHamiltonian, bond::Int) = begin
	((bond >= 1) && (bond <= length(h))) || throw(BoundsError(h.data, bond))
	(bond == length(h)) ? dim(space_r(h)) : dim(⊕(h[bond].rightspaces...))
end 
bond_dimensions(h::MPOHamiltonian) = [bond_dimension(h, i) for i in 1:length(h)]
bond_dimension(h::MPOHamiltonian) = maximum(bond_dimensions(h))



# """
# 	r_RR, right boundary 2-tensor
# 	i-1
# 	o-2
# """
# r_RR(state::MPOHamiltonian) = isomorphism(eltype(state), space_r(state), space_r(state))
# """
# 	l_LL, left boundary 2-tensor
# 	o-1
# 	i-2
# """
# l_LL(state::MPOHamiltonian) = isomorphism(eltype(state), space_l(state), space_l(state))

physical_spaces(x::MPOHamiltonian) = [x[i].pspace for i in 1:length(x)]
isstrict(x::MPOHamiltonian) = isoneunit(space_l(x)) && isoneunit(space_r(x))

"""
	MPO(h::MPOHamiltonian, L::Int) 
	
Conversion of an MPOHamiltonian into a standard MPO
"""
function MPO(h::MPOHamiltonian) 
	L = length(h)
	(L >= 2) || throw(ArgumentError("size of MPO must at least be 2"))
	# isstrict(h) || throw(ArgumentError("only strict MPOHamiltonian is allowed"))
	T = scalartype(h)
	S = spacetype(h)

	mpotensors = Vector{mpotensortype(S, T)}(undef, L)
	embedders = [right_embedders(T, h[i].rightspaces...) for i in 1:length(h)]

	tmp = TensorMap(zeros, T, oneunit(S)*h[1].pspace ← space(embedders[1][1], 2)' * h[1].pspace )
	for i in 1:length(embedders[1])
		@tensor tmp[-1, -2, -3, -4] += h[1, 1, i][-1,-2,1,-4] * embedders[1][i][1, -3]
	end
	mpotensors[1] = tmp
	for n in 2:L-1
		tmp = TensorMap(zeros, T, space(mpotensors[n-1], 3)' * h[n].pspace ← space(embedders[n][1], 2)' * h[n].pspace )
		for (i, j) in opkeys(h[n])
			@tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * h[n, i, j][1,-2,2,-4] * embedders[n][j][2, -3]
		end
		for (i, j) in scalkeys(h[n])
			# iden = h[n].Os[i, j] * isomorphism(Matrix{T}, h[n].pspace, h[n].pspace)
			# @tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * embedders[n][j][1, -3] * iden[-2, -4] 
			@tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * h[n, i, j][1,-2,2,-4] * embedders[n][j][2, -3]
		end
		mpotensors[n] = tmp
	end
	tmp = TensorMap(zeros, T, space(embedders[L-1][1], 2)' * h[L].pspace, space_r(h)' * h[L].pspace )
	_a = size(h[L], 2)
	for i in 1:size(h[L], 1)
		@tensor tmp[-1, -2, -3, -4] += conj(embedders[L-1][i][1, -1]) * h[L, i, _a][1,-2,-3,-4]
	end
	mpotensors[L] = tmp
	return MPO(mpotensors)
end
