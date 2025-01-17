"""
	struct ExactMPS{M<:MPSTensor}
Quantum state exactly represented as MPS without any loss of precision. The center site tensor contains all the 
information and all the other site tensors are simply isomorphism.
"""
struct ExactMPS{M<:MPSTensor} <: AbstractFiniteMPS{M}
	data::Vector{M}
	center::Int

function ExactMPS(data::Vector{M}, center::Int) where {M<:MPSTensor}
	@assert !isempty(data)
	@assert 1 <= center <= length(data)
	new{M}(data, center)
end

end

storage(a::ExactMPS) = a.data
Base.length(a::ExactMPS) = length(storage(a))
Base.isempty(a::ExactMPS) = isempty(storage(a))
Base.getindex(a::ExactMPS, i::Int) = getindex(storage(a), i)
Base.firstindex(a::ExactMPS) = firstindex(storage(a))
Base.lastindex(a::ExactMPS) = lastindex(storage(a))

function Base.setindex!(psi::ExactMPS, v, i::Int)
	# check_mpstensor_dir(v) || throw(SpaceMismatch())
	(i == psi.center) || throw(ArgumentError("only center can be set for ExactMPS"))
	space(v) == space(psi[i]) || throw(SpaceMismatch())
	return setindex!(psi.data, v, i)
end 
Base.copy(psi::ExactMPS) = ExactMPS(copy(psi.data), psi.center)

function Base.complex(psi::ExactMPS)
	if scalartype(psi) <: Real
		data = [complex(item) for item in psi.data]
		return ExactMPS(data, psi.center)
	end
	return psi
end

MPS(psi::ExactMPS) = MPS(psi.data)
ExactMPS(x::ExactMPS) = ExactMPS(x.data, x.center)

"""
	ExactMPS(f, ::Type{T}, physpaces::Vector{S}; left::S=oneunit(S), right::S=oneunit(S))
	ExactMPS(psi::MPS)
constructors of ExactMPS.
1) From a given sector.
2) From a finite MPS.
3) From a density operator in MPS form.
"""
function ExactMPS(f, ::Type{T}, physpaces::Vector{S}; left::S=oneunit(S), right::S=oneunit(S)) where {T <:Number, S <: ElementarySpace}
	mpstensors, left, right, middle_site = _exactmps_side_tensors(T, physpaces, left, right)
	mpstensors[middle_site] = TensorMap(f, T, left ⊗ physpaces[middle_site], right)
	(norm(mpstensors[middle_site]) == 0.) && throw(ArgumentError("invalid sector"))
	return ExactMPS(mpstensors, middle_site)
end

function ExactMPS(psi::MPS)
	target_psi, left, right, middle_site = _exactmps_side_tensors(scalartype(psi), physical_spaces(psi), space_l(psi), space_r(psi)')

	L = length(psi)

	cleft = l_LL(psi)
	cright = r_RR(psi)
	for i in 1:middle_site-1
		cleft = updateleft(cleft, target_psi[i], psi[i])
	end
	for i in L:-1:middle_site+1
		cright = updateright(cright, target_psi[i], psi[i])
	end
	target_psi[middle_site] = ac_prime(psi[middle_site], cleft, cright)
	return ExactMPS([target_psi...], middle_site)
end


function _exactmps_side_tensors(::Type{T}, physpaces::Vector{S}, left::S, right::S) where {T <:Number, S <: ElementarySpace}
	L = length(physpaces)
	middle_site = _find_center(max_bond_dimensions(physpaces, left, right))
	mpstensors = Vector{mpstensortype(S, T)}(undef, L)
	for i in 1:middle_site-1
		physpace = physpaces[i]
		mpstensors[i] = isomorphism(T, left ⊗ physpace, fuse(left, physpace))
		left = space(mpstensors[i], 3)'
	end
	for i in L:-1:middle_site+1
		physpace = physpaces[i]
		tmp = isomorphism(T, fuse(physpace', right), physpace' ⊗ right)
		mpstensors[i] = permute(tmp, (1, 2), (3,))
		right = space(mpstensors[i], 1)
	end
	return mpstensors, left, right, middle_site
end

function _find_center(Ds::Vector{Int})
	pos = argmax(Ds)
	if (pos != 1) && (Ds[pos-1] >= Ds[pos+1])
		pos = pos - 1
	end
	return pos
end
