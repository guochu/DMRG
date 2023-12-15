# It is nontrivial to give a reasonable random MPS ansatz in presence of symmetry

function MPS(f, ::Type{T}, physpaces::Vector{S}, virtualpaces::Vector{S}) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	(length(virtualpaces) == L+1) || throw(DimensionMismatch())
	any(x -> dim(x)==0, virtualpaces) &&  @warn "auxiliary space is empty"
	data = [TensorMap(f, T, virtualpaces[i] ⊗ physpaces[i] ← virtualpaces[i+1]) for i in 1:L]
	return MPS(data)
end

function MPS(f, ::Type{T}, physpaces::Vector{S}, maxvirtualspace::S; left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = infimum(fuse(virtualpaces[i-1], physpaces[i-1]), maxvirtualspace)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1]))
	end
	return MPS(f, T, physpaces, virtualpaces)
end


# for symmetric state
"""
	prodmps(::Type{T}, physpaces::Vector{S}, physectors::Vector; left::S=oneunit(S), right::S=oneunit(S))
Return a product state in MPS form 

Arguments:
* physpaces: the physical spaces on each site.
* physectors: the nonzero physical sectors on each site.

Keyward arguments:
* left: the space of the left most index, in principle this should be vacuum
* right: the space of the left most index, the default is vacuum

The other variants of this functions:
* prodmps(::Type{T}, physpace::S, physectors::Vector; kwargs...): physpace 
shared by all sites
* prodmps(physpaces::Vector{S}, physectors::Vector; kwargs...)
* prodmps(physpace::S, physectors::Vector; kwargs...) 

Initializing a non-symmetric product MPS:
* prodmps(::Type{T}, physpaces::Vector{S}, physectors::Vector{Vector{T2}})
* prodmps(physpaces::Vector{S}, physectors::Vector{Vector{T}})
* prodmps(::Type{T}, physectors::Vector{Vector{T2}})
* prodmps(physectors::Vector{Vector{T}})
* prodmps(::Type{T}, ds::Vector{Int}, physectors::Vector{Int})
* prodmps(ds::Vector{Int}, physectors::Vector{Int}) 
"""
function prodmps(::Type{T}, physpaces::Vector{S}, physectors::Vector; left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	(L == length(physectors)) || throw(DimensionMismatch())
	physectors = [convert(sectortype(S), item) for item in physectors]

	# the total quantum number is ignored in the Abelian case
	if FusionStyle(sectortype(S)) isa UniqueFusion
		rightind, = ⊗(physectors...)
		right = S((rightind=>1,))
	end
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = fuse(virtualpaces[i-1], S((physectors[i-1]=>1,)) )
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(virtualpaces[i+1],  S((physectors[i]=>1,))' ))
	end
	return MPS(ones, T, physpaces, virtualpaces)
end
prodmps(::Type{T}, physpace::S, physectors::Vector; kwargs...) where {T <: Number, S <: ElementarySpace} = prodmps(T, [physpace for i in 1:length(physectors)], physectors; kwargs...)
prodmps(physpaces::Vector{S}, physectors::Vector; kwargs...) where {S <: ElementarySpace} = prodmps(Float64, physpaces, physectors; kwargs...)
prodmps(physpace::S, physectors::Vector; kwargs...) where {S <: ElementarySpace} = prodmps(Float64, physpace, physectors; kwargs...)



"""
	randommps(::Type{T}, physpaces::Vector{S}; D::Int, left::S, right::S)

Return a random MPS.

Each virtual space has multiplicity 1 to allow the largest number of different 
quantum numbers to be explored under bond dimension D
"""
function randommps(::Type{T}, physpaces::Vector{S}; D::Int, left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	virtualpaces = _max_mps_virtual_spaces(physpaces, D, left, right)
	any(x -> dim(x)==0, virtualpaces) &&  @warn "auxiliary space is empty"
	L = length(physpaces)
	A = mpstensortype(S, T)
	mpstensors = Vector{A}(undef, L)
	trunc = truncdim(D)
	for i in 1:L
		virtualpaces[i+1] = infimum(fuse(virtualpaces[i] ⊗ physpaces[i]), virtualpaces[i+1])
		tmp = TensorMap(randn, T, virtualpaces[i] ⊗ physpaces[i] ← virtualpaces[i+1])
		u, s, v = stable_tsvd!(tmp, trunc=trunc)
		mpstensors[i] = u
		virtualpaces[i+1] = space(mpstensors[i], 3)'
	end
	r = MPS(mpstensors)
	rightorth!(r, alg=Orthogonalize(SVD(), trunc, normalize=true))
	return r
end
randommps(physpaces::Vector{S}; kwargs...) where {S <: ElementarySpace} = randommps(Float64, physpaces; kwargs...)

function _max_mps_virtual_spaces(physpaces::Vector{S}, D::Int, left::S=oneunit(S), right::S=oneunit(S)) where {S <: ElementarySpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		tmp = fuse(virtualpaces[i-1], physpaces[i-1])
		virtualpaces[i] = S(Dict(s=>1 for s in sectors(tmp)))
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		tmp = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1]))
		d = max(ceil(Int, D/length(sectors(tmp))), 1)
		virtualpaces[i] = S(Dict(s=>d for s in sectors(tmp)))
	end
	return virtualpaces
end
