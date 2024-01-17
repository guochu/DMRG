# simply store the 3-dimensional tensors and the singular matrices on the bonds.
# It is the user's resonse to make it canonical 
# We will always use the (approximate) right canonical form, in DMRG the mixed form will be handled internally,
# after each sweep the resulting mps will be right canonical


"""
	struct MPS{A<:MPSTensor, B<:MPSBondTensor}

Finite Matrix Product States, which stores a chain of rank-3 tensors (site tensors) 
and another chain of rank-2 tensors (bond tensors).

This is supposed to represent a unitary state

The singular vectors are stored anyway even if the mps is not unitary.
The user is to make sure that they are the correct Schmidt numbers when used

The number of bond tensors is equal to the number of site tensors plus 1, with two trivial 
bond tensors are added on the boundaries.

In most of the algorithms implemented based on MPS, the site tensors are often kept right
-canonical, and the bond tensors contain the Schmidt number of the bipartition.
"""
struct MPS{A<:MPSTensor, B<:MPSBondTensor} <: AbstractFiniteMPS{A}
	data::Vector{A}
	svectors::Vector{Union{Missing, B}}

function MPS(data::Vector{A}, svectors::Vector{Union{Missing, B}}) where {A<:MPSTensor, B<:MPSBondTensor}
	check_mps_spaces(data)
	return new{A, B}(data, svectors)
end

function MPS(data::Vector{A}, svectors::Vector{B}) where {A<:MPSTensor, B<:MPSBondTensor}
	check_mps_spaces(data, svectors)
	return new{A, B}(data, convert(Vector{Union{Missing, B}}, svectors))
end

"""
	MPS{A, B}(mpstensors::Vector)

Return a general MPS from a chain of rank-3 tensors, with singular vectors 
left uninitialized. Main entrance for constructing a finite MPS

Convention of the site tensors used (i mean in arrow, o means out arrow):
    o 
    |
o-1 2 3-i
The quantum number flows from right to left.	

In principle the left-most index (space_l(mps)) and the right-most index (space_r(mps))
can be arbitrary. However for most algorithms it is assumed (and asserted) that 
'space_l(mps) = vacuum'. An MPS could also have multiple sectors, but for certain
algorithms such as DMRG, only one sector is allowed.

Other constructors:
* MPS(f, ::Type{T}, physpaces::Vector{S}, virtualpaces::Vector{S}) 
* MPS(f, ::Type{T}, physpaces::Vector{S}, maxvirtualspace::S; left::S=oneunit(S), right::S=oneunit(S)) 
"""
function MPS(data::Vector{A}) where {A<:MPSTensor}
	check_mps_spaces(data)	
	T = real(scalartype(A))
	B = bondtensortype(spacetype(A), Diagonal{T, Vector{T}})
	svectors = Vector{Union{Missing, B}}(missing, length(data)+1)
	svectors[1] = Diagonal(id(space_l(data[1])))
	svectors[end] = Diagonal(id(space_r(data[end])'))
	return new{A, B}(data, svectors)
end 

end


function check_mps_spaces(data::AbstractVector)
	@assert !isempty(data)
	# all(check_mpstensor_dir, data) || throw(SpaceMismatch())
	isoneunit(space_l(data[1])) || throw(SpaceMismatch("space_l of MPS should be vacuum by convention"))
	for i in 1:length(data)-1
		(space_r(data[i])' == space_l(data[i+1])) || throw(SpaceMismatch())
	end
end
function check_mps_spaces(data::AbstractVector, svectors::AbstractVector)
	@assert length(data)+1 == length(svectors)
	(isoneunit(space_l(data[1])) && isoneunit(space(svectors[1], 1))) || throw(SpaceMismatch("space_l of MPS should be vacuum by convention"))
	for i in 1:length(data)-1
		(space_r(data[i])' == space_l(data[i+1]) == space(svectors[i+1], 1)) || throw(SpaceMismatch())
	end	
	(space(svectors[end], 1) == space_r(data[end])') || throw(SpaceMismatch())
end

bondtensortype(::Type{MPS{A, B}}) where {A, B} = B
bondtensortype(a::MPS) = bondtensortype(typeof(a))

function Base.getproperty(psi::MPS, s::Symbol)
	if s == :s
		return MPSBondView(psi)
	else
		return getfield(psi, s)
	end
end

storage(a::MPS) = a.data
Base.length(a::MPS) = length(storage(a))
Base.isempty(a::MPS) = isempty(storage(a))
Base.getindex(a::MPS, i::Int) = getindex(storage(a), i)
Base.firstindex(a::MPS) = firstindex(storage(a))
Base.lastindex(a::MPS) = lastindex(storage(a))


function Base.setindex!(psi::MPS, v::MPSTensor, i::Int)
	# check_mpstensor_dir(v) || throw(SpaceMismatch())
	if i == 1
		isoneunit(space_l(v)) || throw(SpaceMismatch("space_l of MPS should be vacuum by convention."))
	end
	return setindex!(psi.data, v, i)
end 
Base.copy(psi::MPS) = MPS(copy(psi.data), copy(psi.svectors))

# function Base.vcat(psiA::MPS, psiB::MPS)
# 	(space_r(psiA)' == space_l(psiB)) || throw(SpaceMismatch("cannot cat two states with incompatible sectors."))
# 	return MPS(vcat(psiA.data, psiB.data))
# end
function Base.complex(psi::MPS)
	if scalartype(psi) <: Real
		data = [complex(item) for item in psi.data]
		return MPS(data, psi.svectors)
	end
	return psi
end

svectors_uninitialized(psi::MPS) = any(ismissing, psi.svectors)
function unset_svectors!(psi::MPS)
	psi.svectors[2:end-1] .= missing
	return psi
end

"""
	isrightcanonical(a::MPS; kwargs...)

Return if an MPS is in right canonical form
"""
isrightcanonical(a::MPS; kwargs...) = all(x->isrightcanonical(x; kwargs...), a.data)
function isleftcanonical(a::MPS; kwargs...) 
	all(x->isleftcanonical(x; kwargs...), a.data[1:end-1]) || return false
	# return isleftcanonical(a.data[end] * sqrt(dim(space_r(a))); kwargs...)
	return isleftcanonical_r(a.data[end]; kwargs...)
end

"""
	iscanonical(psi::MPS; kwargs...) 

Check if the MPS is right canonical, and whether the singular values are correct
"""
function iscanonical(a::MPS; kwargs...)
	isrightcanonical(a) || return false
	# we also check whether the singular vectors are the correct Schmidt numbers
	svectors_uninitialized(a) && return false
	hold = l_LL(a)
	for i in 1:length(a)-1
		hold = updateleft(hold, a[i], a[i])
		tmp = a.s[i+1] * a.s[i+1]
		isapprox(hold, tmp; kwargs...) || return false
	end
	return true
end

function max_bond_dimensions(physpaces::Vector{S}, left::S, right::S) where {S <: ElementarySpace}
	L = length(physpaces)
	# left = oneunit(S)
	# right = S(sector=>1)
	Ds = [1 for i in 1:L+1]
	for i in 1:L
		left = fuse(left, physpaces[i])
		Ds[i+1] = dim(left)
	end
	Ds[end] = 1
	for i in L:-1:1
		right = fuse(physpaces[i]', right)
		Ds[i] = min(Ds[i], dim(right))
	end
	return Ds
end
# for general MPS this is the only we can do
function max_bond_dimensions(psi::MPS)
	isstrict(psi) || throw(ArgumentError("only strict mps allowed"))
	return max_bond_dimensions(physical_spaces(psi), space_l(psi), space_r(psi))
end

function max_bond_dimensions(physpaces::Vector{Int}, D::Int) 
	L = length(physpaces)
	left = 1
	right = 1
	virtualpaces = Vector{Int}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = min(virtualpaces[i-1] * physpaces[i-1], D)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = min(virtualpaces[i], physpaces[i] * virtualpaces[i+1])
	end
	return virtualpaces
end
