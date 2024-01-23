struct OverlapTransferMatrix{M <: Union{MPSTensor, MPOTensor}}
	above::Vector{M}
	below::Vector{M}
end

Base.length(x::OverlapTransferMatrix) = length(x.above)
TK.scalartype(::Type{OverlapTransferMatrix{M}}) where M = scalartype(M)

function Base.:*(left::MPSBondTensor, m::OverlapTransferMatrix)
	for (a, b) in zip(m.above, m.below)
		left = updateleft(left, a, b)
	end
	return left
end

function Base.:*(m::OverlapTransferMatrix, right::MPSBondTensor)
	for (a, b) in Iterators.reverse(zip(m.above, m.below))
		right = updateright(right, a, b)
	end
	return right
end
 
function TransferMatrix(x::Vector{M}, y::Vector{M}) where {M <: Union{MPSTensor, MPOTensor}}
	@assert length(x) == length(y)
	return OverlapTransferMatrix(x, y)
end

# noticing the index order convention!!!
function Base.convert(::Type{<:TensorMap}, x::OverlapTransferMatrix)
	L = length(x)
	@tensor left[1,4;3,5] := conj(x.above[1][1,2,3]) * x.below[1][4,2,5]
	for i in 2:L
		left = @tensor tmp[1,2;6,7] := left[1,2,3,4] * conj(x.above[i][3,5,6]) * x.below[i][4,5,7]
	end
	return left
end

struct ExpecTransferMatrix{M <: MPOTensor, V <: Union{MPSTensor, MPOTensor}}
	above::Vector{V}
	middle::Vector{M}
	below::Vector{V}
end

Base.length(x::ExpecTransferMatrix) = length(x.above)
TK.scalartype(::Type{ExpecTransferMatrix{M, V}}) where {M, V} = promote_type(scalartype(M), scalartype(V))

function Base.:*(left::MPSTensor, m::ExpecTransferMatrix)
	for (a, h, b) in zip(m.above, m.middle, m.below)
		left = updateleft(left, a, h, b)
	end
	return left
end
function Base.:*(m::ExpecTransferMatrix, right::MPSTensor)
	for (a, h, b) in Iterators.reverse(zip(m.above, m.middle, m.below))
		right = updateright(right, a, h, b)
	end
	return right	
end

function TransferMatrix(above::Vector{V}, middle::Vector{<:MPOTensor}, below::Vector{V}) where {V <: Union{MPSTensor, MPOTensor}}
	@assert length(above) == length(middle) == length(below)
	return ExpecTransferMatrix(above, middle, below)
end

