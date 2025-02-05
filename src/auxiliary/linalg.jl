

"""
	renyi_entropy(v::AbstractVector{<:Real}; α::Real=1) 

Compute the renyi entropy of a vector "v"
Requirement: sum of v should be 1
"""
function renyi_entropy(v::AbstractVector{<:Real}; α::Real=1) 
	α = convert(eltype(v), α)
	a = _check_and_filter(v)
	if α==one(α)
	    return -dot(a, log.(a))
	else
		a = v.^(α)
		return (1/(1-α)) * log(sum(a))
	end
end


function _check_and_filter(v::AbstractVector{<:Real}; tol::Real=1.0e-12)
	(abs(sum(v) - 1) <= tol) || throw(ArgumentError("sum of singular values not equal to 1"))
	oo = zero(eltype(v))
	tol = convert(eltype(v), tol)
	for item in v
		((item < oo) && (-item > tol)) && throw(ArgumentError("negative singular values"))
	end
	# return [(abs(item) <= tol) ? oo : item for item in v]
	return [item for item in v if abs(item) > tol] 
end

function _distance2(x, y)
	sA = real(dot(x, x))
	sB = real(dot(y, y))
	c = dot(x, y)
	r = sA+sB-2*real(c)
	return abs(r)
end

_distance(x, y) = sqrt(_distance2(x, y))


function entanglement_spectrum(m::AbstractTensorMap)
	u, ss, v, err = stable_tsvd(m, trunc = TK.NoTruncation())
	return LinearAlgebra.diag(convert(Array,ss))
end

entanglement_spectrum(m::DiagonalTensorMap) = m.data


function stable_tsvd(m::AbstractTensorMap, args...; trunc::TruncationScheme=NoTruncation())
	try
		return tsvd(m, args...; trunc=trunc, alg=TK.SDD())
	catch
		return tsvd(m, args...; trunc=trunc, alg=TK.SVD())
	end
end

function stable_tsvd!(m::AbstractTensorMap; trunc::TruncationScheme=NoTruncation())
	try
		return tsvd!(copy(m), trunc=trunc, alg=TK.SDD())
	catch
		return tsvd!(m, trunc=trunc, alg=TK.SVD())
	end
end


loose_isometry(cod::TensorSpace, dom::TensorSpace) =
    loose_isometry(Matrix{Float64}, cod, dom)
loose_isometry(P::TensorSpace) = loose_isometry(codomain(P), domain(P))
loose_isometry(A::Type{<:DenseMatrix}, P::TensorSpace) =
    loose_isometry(A, codomain(P), domain(P))
function loose_isometry(::Type{A},
                    cod::TensorSpace,
                    dom::TensorSpace) where {A<:DenseMatrix}
    t = TensorMap{scalartype(A)}(undef, cod, dom)
    for (c, b) in blocks(t)
        TK.MatrixAlgebra.one!(b)
    end
    return t
end

function right_embedders(::Type{T}, a::S...) where {T <: Number, S <: ElementarySpace}
    V = ⊕(a...) 
    ts = [zeros(T, aj, V) for aj in a]
    for c in sectors(V)
    	n = 0
    	for i in 1:length(ts)
    		ni = dim(a[i], c)
    		block(ts[i], c)[:, (n+1):(n+ni)] .= Diagonal( ones(ni) )
    		n += ni
    	end
    end
    return ts
end

function left_embedders(::Type{T}, a::S...) where {T <: Number, S <: ElementarySpace}
    V = ⊕(a...) 
    ts = [zeros(T, V, aj) for aj in a]
    for c in sectors(V)
    	n = 0
    	for i in 1:length(ts)
    		ni = dim(a[i], c)
    		block(ts[i], c)[(n+1):(n+ni), :] .= Diagonal( ones(ni) )
    		n += ni
    	end
    end
    return ts	
end


function Base.cat(t1::AbstractTensorMap{<:Number, S, N₁, N₂}, t2::AbstractTensorMap{<:Number, S, N₁, N₂}; dims::Integer) where {S, N₁, N₂}
    dims_rest = tuple((x for x in 1:N₁+N₂ if x != dims)...)
    t1′ = permute(t1, (dims,), dims_rest)
    t2′ = permute(t2, (dims,), dims_rest)
    t = catcodomain(t1′, t2′)
    perm = (dims, dims_rest...)
    iperm = TupleTools.invperm(perm)
    p1 = iperm[1:N₁]
    p2 = iperm[N₁+1:N₁+N₂]
    return permute(t, p1, p2)
end

function copytodiagonal!(x, y)
	for i in 1:size(x, 1)
		x[i, i] = y[i, i]
	end
	return x
end

# some patches
function LinearAlgebra.Diagonal(t::AbstractTensorMap{<:Number, <:IndexSpace, 1, 1})
	r = DiagonalTensorMap{scalartype(t)}(undef, domain(t)[1])
	for (k, v) in blocks(t)
		copytodiagonal!(block(r, k), v)
	end
	return r
end 
TK.:⊗(::Type{I}) where {I<:Sector} = (one(I),)
TK.:⊗(::Type{I}, a::I, rest::Vararg{I}) where {I<:Sector} = ⊗(a, rest...)


# function _one!(A::AbstractMatrix)
#     for j in 1:size(A, 2)
#         for i in 1:size(A, 1)
#             @inbounds A[i, j] = i == j
#         end
#     end
#     return A
# end