
# customized truncation

"""
    struct TruncationDimCutoff
truncate singular values below ϵ first, if the remaining bond dimension is larger than dim, then truncate it below dim.
Return the p-norm of the truncated singular values.
"""
struct TruncationDimCutoff <: TruncationScheme
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
        init += dim(c) * length(b)
    end
    return init
end

function TK._compute_truncdim(Σdata, trunc::TruncationDimCutoff, p=2)
    n = TK._norm(Σdata, p, 0.)
    truncdim1 = TK._compute_truncdim(Σdata, truncbelow(trunc.ϵ * n, trunc.add_back), p)
    if compute_size(truncdim1) <= trunc.D
        return truncdim1
    end
    return TK._compute_truncdim(Σdata, truncdim(trunc.D), p)
end