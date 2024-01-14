# another MPO object is needed to support non-number conserving operators, such a

"""
	PartialMPO{A <: MPOTensor}
Finite Matrix Product Operator which stores a chain of rank-4 site tensors.
"""
struct PartialMPO{A <: MPOTensor} <: AbstractMPO{A}
	data::Vector{A}
	positions::Vector{Int}

"""
	PartialMPO{A}(mpotensors::Vector, positions::Vector{Int})
Constructor entrance for MPO, which only supports strictly quantum number conserving operators

Site tensor convention:
i mean in arrow, o means out arrow
    o 
    |
    2
o-1   3-i
	4
	|
	i
The left and right boundaries are always vacuum.
The case that the right boundary is not vacuum corresponds to operators which do 
not conserve quantum number, such as a†.
"""
function PartialMPO(data::AbstractVector{A}, positions::AbstractVector{Int}) where {A<:MPOTensor}
	@assert !isempty(data)
	@assert length(data) == length(positions)
	# @assert positions[1] > 0 # allow periodic condition
	check_mpo_spaces(data)
	isoneunit(space_r(data[end])) || throw(ArgumentError("only strict PartialMPO allowed"))
	for i in 1:length(positions)-1
		(positions[i] < positions[i+1]) || throw(ArgumentError("positions should be ordered from small to large"))
	end
	return new{A}(convert(Vector{A}, data), convert(Vector{Int}, positions))
end
end

storage(a::PartialMPO) = a.data
positions(a::PartialMPO) = a.positions
function Base.setindex!(h::PartialMPO, v::MPOTensor, i::Int)
	# check_mpotensor_dir(v) || throw(SpaceMismatch())
	if i == 1
		isoneunit(space_l(v)) || throw(SpaceMismatch("space_l of the left boundary tensor should be vacuum by convention"))
	end
	return setindex!(h.data, v, i)
end 
Base.copy(h::PartialMPO) = PartialMPO(copy(storage(h)), copy(positions(h)))

function Base.complex(psi::PartialMPO)
	if scalartype(psi) <: Real
		data = [complex(item) for item in psi.data]
		return PartialMPO(data, positions(psi))
	end
	return psi
end

bond_dimension(h::PartialMPO, bond::Int) = begin
	((bond >= 1) && (bond <= length(h))) || throw(BoundsError(storage(h), bond))
	dim(space(h[bond], 3))
end 

isrightcanonical(a::PartialMPO; kwargs...) = all(x->isrightcanonical(x; kwargs...), a.data)
function isleftcanonical(a::PartialMPO; kwargs...) 
	all(x->isleftcanonical(x; kwargs...), a.data[1:end-1]) || return false
	# return isleftcanonical(a.data[end] * sqrt(dim(space_r(a))); kwargs...)
	return isleftcanonical_r(a.data[end]; kwargs...)
end



