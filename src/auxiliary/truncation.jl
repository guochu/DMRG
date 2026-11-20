
# customized truncation

"""
    struct TruncationDimCutoff
truncate singular values below ϵ first, if the remaining bond dimension is larger than dim, then truncate it below dim.
Return the p-norm of the truncated singular values.

The threshold ϵ is relative to the p-norm of the full spectrum, and the comparison between
sectors is weighted by the quantum dimensions, as in the truncation schemes of TensorKit.
"""
struct TruncationDimCutoff <: TruncationStrategy
    D::Int
    ϵ::Float64
    add_back::Int
end
TruncationDimCutoff(;D::Int, ϵ::Real, add_back::Int=0) = TruncationDimCutoff(D, convert(Float64, ϵ), min(add_back, D))
truncdimcutoff(D::Int, epsilon::Real; add_back::Int=0) = TruncationDimCutoff(D, epsilon, add_back)
truncdimcutoff(; D::Int, ϵ::Real, add_back::Int=0) = TruncationDimCutoff(D, convert(Float64, ϵ), min(add_back, D))

compute_size(v::AbstractVector) = length(v)
function compute_size(v::AbstractDict)
    init = 0
    for (c, b) in v
        init += dim(c) * b
    end
    return init
end

# dim-weighted p-norm of the spectrum
function _weighted_pnorm(values::SectorVector, p::Real)
    nᵖ = mapreduce(+, pairs(values); init=zero(real(eltype(values)))) do (c, v)
        dim(c) * mapreduce(x -> abs(x)^p, +, v; init=zero(real(eltype(v))))
    end
    return nᵖ^inv(p)
end

# sector holding the smallest kept singular value (dim-weighted), values sorted descending per sector
function _findnexttruncvalue(values::SectorVector, truncdim::SectorDict{I,Int}, p::Real) where {I<:Sector}
    (isempty(values) || all(iszero, Base.values(truncdim))) && return nothing

    cmin = first(c for (c, d) in truncdim if d > 0)
    weightedσmin = dim(cmin)^inv(p) * values[cmin][truncdim[cmin]]
    for (c, σs) in pairs(values)
        if truncdim[c] > 0
            weightedσ = dim(c)^inv(p) * σs[truncdim[c]]
            if weightedσ < weightedσmin
                cmin, weightedσmin = c, weightedσ
            end
        end
    end
    return cmin
end

# sector holding the largest discarded singular value (dim-weighted)
function _findnextgrowvalue(values::SectorVector, truncdim::SectorDict{I,Int}, p::Real) where {I<:Sector}
    istruncated = SectorDict{I,Bool}(c => (d < length(values[c])) for (c, d) in truncdim)
    (isempty(values) || !any(Base.values(istruncated))) && return nothing

    cmax = first(c for (c, s) in istruncated if s)
    weightedσmax = dim(cmax)^inv(p) * values[cmax][truncdim[cmax] + 1]
    for (c, σs) in pairs(values)
        if istruncated[c]
            weightedσ = dim(c)^inv(p) * σs[truncdim[c] + 1]
            if weightedσ > weightedσmax
                cmax, weightedσmax = c, weightedσ
            end
        end
    end
    return cmax
end

function MAK.findtruncated(values::SectorVector, trunc::TruncationDimCutoff)
    I = sectortype(values)
    p = 2.0
    n = _weighted_pnorm(values, p)
    # keep all singular values above the relative cutoff ϵ * n
    truncdim = SectorDict{I,Int}()
    for (c, v) in pairs(values)
        d = findlast(Base.Fix2(>, trunc.ϵ * n), v)
        truncdim[c] = isnothing(d) ? 0 : d
    end
    # add back discarded singular values, largest first
    for _ in 1:trunc.add_back
        cmax = _findnextgrowvalue(values, truncdim, p)
        isnothing(cmax) && break
        truncdim[cmax] += 1
    end
    # cap the total (dim-weighted) bond dimension
    if compute_size(truncdim) > trunc.D
        while compute_size(truncdim) > trunc.D
            cmin = _findnexttruncvalue(values, truncdim, p)
            isnothing(cmin) && break
            truncdim[cmin] -= 1
        end
    end
    return SectorDict{I, UnitRange{Int}}(c => (1:d) for (c, d) in truncdim)
end
