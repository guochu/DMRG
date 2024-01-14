


struct AdjointMPO{A <: MPOTensor}
	parent::MPO{A}
end


Base.length(h::AdjointMPO) = length(h.parent)
TK.scalartype(h::Type{AdjointMPO{A}}) where A = scalartype(A)

Base.adjoint(h::MPO) = AdjointMPO(h)
Base.adjoint(h::AdjointMPO) = h.parent

Base.getindex(a::AdjointMPO, i::Int) = mpotensor_adjoint(getindex(a.parent, i))
Base.firstindex(a::AdjointMPO) = mpotensor_adjoint(firstindex(a.parent))
Base.lastindex(a::AdjointMPO) = mpotensor_adjoint(lastindex(a.parent))


isstrict(h::AdjointMPO) = isstrict(h.parent)

bond_dimension(h::AdjointMPO, bond::Int) = begin
    ((bond >= 1) && (bond <= length(h))) || throw(BoundsError(storage(h.parent), bond))
    dim(space(h.parent[bond], 3))
end 

"""
    Base.convert(::Type{<:MPO}, h::AdjointMPO) 

This implementation may have a problem
"""
function Base.convert(::Type{<:MPO}, h::AdjointMPO) 
    isstrict(h) || throw(ArgumentError("only strict operator allowed"))
    return MPO(unsafe_mpotensor_adjoint.(h.parent.data))
end

"""
    Base.convert(::Type{<:AdjointMPO}, h::MPO)

This implementation may have a problem
"""
function Base.convert(::Type{<:AdjointMPO}, h::MPO) 
    isstrict(h) || throw(ArgumentError("only strict operator allowed"))
    return AdjointMPO(MPO(unsafe_mpotensor_adjoint.(h.data)))
end
function unsafe_mpotensor_adjoint(vj::MPOTensor)
    rj = vj'
    sl = space(rj, 3)'
    ml = isomorphism(Matrix{scalartype(vj)}, sl, flip(sl))
    sr = space(rj, 1)
    mr = isomorphism(Matrix{scalartype(vj)}, flip(sr), sr)
    @tensor tmp[-1 -2; -3 -4] := ml[1, -1] * rj[2,-2,1,-4] * mr[-3,2]
    return tmp
end

mpotensor_adjoint(m::MPOTensor) = permute(m', (3,2), (1,4))