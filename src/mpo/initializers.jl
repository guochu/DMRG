function MPO(f, ::Type{T}, physpaces::Vector{S}, virtualpaces::Vector{S}) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	(length(virtualpaces) == L+1) || throw(DimensionMismatch())
	any(x -> dim(x)==0, virtualpaces) &&  @warn "auxiliary space is empty"
	data = [TensorMap(f, T, virtualpaces[i] ⊗ physpaces[i] ← virtualpaces[i+1] ⊗ physpaces[i] ) for i in 1:L]
	return MPO(data)
end

function MPO(f, ::Type{T}, physpaces::Vector{S}, maxvirtualspace::S; left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = infimum(fuse(virtualpaces[i-1], physpaces[i-1], physpaces[i-1]'), maxvirtualspace)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1], physpaces[i]))
	end
	return MPO(f, T, physpaces, virtualpaces)
end

# generic initializers

"""
	prodmpo(::Type{T}, physpaces::Vector{S}, pos::Vector{Int}, ms::Vector{M})

Return a product mpo using a chain of 4-dimensional tensors, missing tensors are 
interpretted as identity.

Other variants
* prodmpo(physpaces::Vector{S}, pos::Vector{Int}, ms::Vector{M})
* prodmpo(::Type{T}, L::Int, pos::Vector{Int}, ms::Vector{M})
* prodmpo(L::Int, pos::Vector{Int}, ms::Vector{M})
"""
function prodmpo(::Type{T}, physpaces::Vector{S}, positions::Vector{Int}, ops::Vector) where {T <: Number, S <: ElementarySpace}
	@assert all(x->isa(x, SiteOperator), ops)
	@assert issorted(positions)
	L = length(physpaces)
	for (k, v) in zip(positions, ops)
		((k>= 1) && (k <= L)) || throw(BoundsError(physpaces, k))
		(physpaces[k] == ophysical_space(v) == iphysical_space(v)') || throw(SpaceMismatch("space mismatch on site $k"))
	end
	A = mpotensortype(S, T)
	mpotensors = Vector{A}(undef, L)
	left = oneunit(S)
	for i in 1:L
		pos = findfirst(x->x==i, positions)
		if isnothing(pos)
			mj = id(left ⊗ physpaces[i])
		else
			tmp = ops[pos]
			if isa(tmp, MPSBondTensor)
				mj = _add_legs(tmp, left)
			else
				mj = tmp
			end
		end
		mpotensors[i] = convert(A, mj) 
		left = space_r(mpotensors[i])'
	end
	return MPO(mpotensors)
end

prodmpo(physpaces::Vector{S}, pos::Vector{Int}, ms::Vector) where {S <: ElementarySpace} = prodmpo(compute_scalartype(ms), physpaces, pos, ms)

function prodmpo(::Type{T}, L::Int, pos::Vector{Int}, ms::Vector) where {T <: Number}
	isempty(ms) && error("input should not be empty.")
	S = physical_space(ms[1])
	return prodmpo(T, [S for i in 1:L], pos, ms)
end
prodmpo(L::Int, pos::Vector{Int}, ms::Vector) = prodmpo(compute_scalartype(ms), L, pos, ms)
prodmpo(physpaces::Vector{<:ElementarySpace}, m::PartialMPO) = prodmpo(physpaces, positions(m), storage(m))

function compute_scalartype(a)
	T = Float64
	for v in a
		T = promote_type(T, scalartype(v))
	end
	return T
end

function _add_legs(m::AbstractTensorMap{<:Number, S, 1, 1}, left::S) where {S <: ElementarySpace}
	virtual = isomorphism(scalartype(m), left, left)
	@tensor tmp[1,3;2,4] := virtual[1, 2] * m[3,4]
	return tmp
end



"""
	randommpo(::Type{T}, physpaces::Vector{S}; D::Int, left::S, right::S)

Return a random MPO.

Each virtual space has multiplicity 1 to allow the largest number of different 
quantum numbers to be explored under bond dimension D
"""
function randommpo(::Type{T}, physpaces::Vector{S}; D::Int, left::S=oneunit(S), right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	virtualpaces = _max_mpo_virtual_spaces(physpaces, D, left, right)
	any(x -> dim(x)==0, virtualpaces) &&  @warn "auxiliary space is empty"
	L = length(physpaces)
	A = mpotensortype(S, T)
	mpstensors = Vector{A}(undef, L)
	trunc = truncdim(D)
	for i in 1:L
		virtualpaces[i+1] = infimum(fuse(physpaces[i], virtualpaces[i], physpaces[i]'), virtualpaces[i+1])
		tmp = TensorMap(randn, T, virtualpaces[i] ⊗ physpaces[i] ⊗ physpaces[i]' ←  virtualpaces[i+1])
		u, s, v = stable_tsvd!(tmp, trunc=trunc)
		mpstensors[i] = permute(u, (1,2), (4, 3))
		virtualpaces[i+1] = space(mpstensors[i], 3)'
	end
	r = MPO(mpstensors)
	rightorth!(r, alg=Orthogonalize(SVD(), trunc, normalize=true))
	return r
end
randommpo(physpaces::Vector{S}; kwargs...) where {S <: ElementarySpace} = randommpo(Float64, physpaces; kwargs...)

function _max_mpo_virtual_spaces(physpaces::Vector{S}, D::Int, left::S=oneunit(S), right::S=oneunit(S)) where {S <: ElementarySpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		tmp = fuse(virtualpaces[i-1], physpaces[i-1], physpaces[i-1]')
		virtualpaces[i] = S(Dict(s=>1 for s in sectors(tmp)))
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		tmp = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1], physpaces[i])) 
		d = max(ceil(Int, D/length(sectors(tmp))), 1)
		virtualpaces[i] = S(Dict(s=>1 for s in sectors(tmp)))
	end
	return virtualpaces
end
