# another MPO object is needed to support non-number conserving operators, such a

"""
	MPO{A <: MPOTensor}
Finite Matrix Product Operator which stores a chain of rank-4 site tensors.
"""
struct MPO{A <: MPOTensor} <: AbstractMPO{A}
	data::Vector{A}

"""
	MPO{A}(mpotensors::Vector)
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
function MPO(data::Vector{A}) where {A<:MPOTensor}
	@assert !isempty(data)
	check_mpo_spaces(data)
	return new{A}(data)
end
end
coeff(x::MPO) = one(scalartype(x))
# overall_scale(x::AbstractMPO) = coeff(x)^(length(x))
# rescaling!(psi::MPO, n::Real) = _rescaling!(psi, n)
# function normalize_coeff!(x::MPO)
# 	c = coeff(x)
# 	if c != one(c)
# 		for i in 1:length(x)
# 			x[i] = lmul!(c, x[i])
# 		end
# 		x.coeff[] = 1
# 	end
# 	return x
# end

function check_mpo_spaces(mpotensors::AbstractVector)
	# all(check_mpotensor_dir, mpotensors) || throw(SpaceMismatch())
	for i in 1:length(mpotensors)-1
		(space_r(mpotensors[i]) == space_l(mpotensors[i+1])') || throw(SpaceMismatch())
	end
	isoneunit(space_l(mpotensors[1])) || throw(SpaceMismatch("space_l of the left boundary tensor should be vacuum by convention"))
end

storage(a::MPO) = a.data
function Base.setindex!(h::MPO, v::MPOTensor, i::Int)
	# check_mpotensor_dir(v) || throw(SpaceMismatch())
	if i == 1
		isoneunit(space_l(v)) || throw(SpaceMismatch("space_l of the left boundary tensor should be vacuum by convention"))
	end
	return setindex!(h.data, v, i)
end 
Base.copy(h::MPO) = MPO(copy(h.data))

function Base.complex(psi::MPO)
	if scalartype(psi) <: Real
		data = [complex(item) for item in psi.data]
		return MPO(data)
	end
	return psi
end

bond_dimension(h::MPO, bond::Int) = begin
	((bond >= 1) && (bond <= length(h))) || throw(BoundsError(storage(h), bond))
	dim(space(h[bond], 3))
end 

isrightcanonical(a::MPO; kwargs...) = all(x->isrightcanonical(x; kwargs...), a.data)
function isleftcanonical(a::MPO; kwargs...) 
	all(x->isleftcanonical(x; kwargs...), a.data[1:end-1]) || return false
	# return isleftcanonical(a.data[end] * sqrt(dim(space_r(a))); kwargs...)
	return isleftcanonical_r(a.data[end]; kwargs...)
end

# function select_sector(h::MPO; sector::Sector)
# 	s_r = space_r(h)'
# 	hassector(s_r, sector) || throw(ArgumentError("sector does not exist."))
# 	m = isometry(s_r, typeof(s_r)(sector=>dim(s_r, sector))) 
# 	mpotensors = copy(raw_data(h)[1:end-1])
# 	@tensor tmp[-1 -2 ; -3 -4] := h[end][-1, -2, 1, -4] * m[1, -3]
# 	push!(mpotensors, tmp)
# 	return MPO(mpotensors)
# end
