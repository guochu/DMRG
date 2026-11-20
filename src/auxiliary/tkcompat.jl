# Compatibility layer for TensorKit v0.17 (MatrixAlgebraKit-based factorizations).
# Restores the TensorKit v0.13-style convenience API used throughout this package.

import MatrixAlgebraKit as MAK
using MatrixAlgebraKit: TruncationStrategy, NoTruncation, TruncatedAlgorithm
using TensorKit: IndexTuple, SectorVector, SectorDict

# factorization algorithm markers (exported by TensorKit v0.13, removed in v0.14)
struct QR end
struct LQ end
struct SVD end
struct SDD end

# v0.13 called the truncation strategy type `TruncationScheme`
const TruncationScheme = TruncationStrategy

# `permute(t, p₁, p₂)` with two separate index tuples (removed in TensorKit v0.14)
function TK.permute(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...)
    return permute(t, (p₁, p₂); kwargs...)
end

# `TensorMap(f, T, ...)` with a function initializer `f ∈ (rand, randn, zeros, ones)`
# (deprecated in TensorKit v0.13, removed in v0.14)
for f in (:rand, :randn, :zeros, :ones)
    @eval begin
        TK.TensorMap(::typeof($f), T::Type{<:Number}, V::TensorMapSpace) = $f(T, V)
        TK.TensorMap(::typeof($f), T::Type{<:Number}, cod::TensorSpace, dom::TensorSpace) = $f(T, cod, dom)
        TK.TensorMap(::typeof($f), T::Type{<:Number}, cod::TensorSpace) = $f(T, cod)
    end
end

# `leftorth`, `rightorth`, `leftnull`, `rightnull` with optional index permutation;
# `alg` keywords accept the marker types `QR()`, `SVD()`, `LQ()` as in v0.13.
function leftorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...)
    return leftorth(permute(t, (p₁, p₂)); kwargs...)
end
function leftorth(t::AbstractTensorMap; alg::Union{QR, SVD}=QR(), kwargs...)
    return leftorth!(copy(t); alg=alg, kwargs...)
end
function leftorth!(t::AbstractTensorMap; alg::Union{QR, SVD}=QR(), kwargs...)
    return alg isa SVD ? left_orth(t; alg=:svd, kwargs...) : left_orth(t; alg=:qr, kwargs...)
end

function rightorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...)
    return rightorth(permute(t, (p₁, p₂)); kwargs...)
end
function rightorth(t::AbstractTensorMap; alg::Union{LQ, SVD}=LQ(), kwargs...)
    return rightorth!(copy(t); alg=alg, kwargs...)
end
function rightorth!(t::AbstractTensorMap; alg::Union{LQ, SVD}=LQ(), kwargs...)
    return alg isa SVD ? right_orth(t; alg=:svd, kwargs...) : right_orth(t; alg=:lq, kwargs...)
end

function leftnull(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...)
    return leftnull(permute(t, (p₁, p₂)); kwargs...)
end
function leftnull(t::AbstractTensorMap; kwargs...)
    # `left_null` returns the complement `N` with `N' * t = 0`, as in v0.13
    return left_null(t; kwargs...)
end

function rightnull(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...)
    return rightnull(permute(t, (p₁, p₂)); kwargs...)
end
function rightnull(t::AbstractTensorMap; kwargs...)
    # both v0.13 `rightnull` and v0.17 `right_null` return the basis `N`
    # (shaped `W ← domain`) with `t * N' = 0`
    return right_null(t; kwargs...)
end

# `scalar` of a fully contracted TensorMap with all spaces oneunit
# (TensorKit v0.13 accepted any rank, v0.17 only accepts rank-0 tensors)
function TK.scalar(t::AbstractTensorMap)
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
           first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end

# truncated SVD with the robust-algorithm fallback of `stable_tsvd`
function stable_tsvd(m::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...)
    return stable_tsvd(permute(m, (p₁, p₂)); kwargs...)
end
function stable_tsvd(m::AbstractTensorMap; trunc::TruncationScheme=notrunc())
    try
        return svd_trunc(m; trunc=trunc)
    catch
        # robust fallback for LAPACK DivideAndConquer convergence failures,
        # analogous to the `SDD()` -> `SVD()` fallback of TensorKit v0.13
        return svd_trunc(m, TruncatedAlgorithm(MAK.LAPACK_QRIteration(), trunc))
    end
end
stable_tsvd!(m::AbstractTensorMap; kwargs...) = stable_tsvd(m; kwargs...)
