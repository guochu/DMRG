"""
    fromABCD(;B::AbstractVector, C::AbstractVector, D::Union{MPSBondTensor, Number}=1, A::AbstractMatrix=zeros(length(B), length(C)))

Return an MPOTensor of the following form
1 C D
0 A B
0 0 1
"""
function fromABCD(;B::AbstractVector, C::AbstractVector, D::Union{MPSBondTensor, Number}=1, A::AbstractMatrix=zeros(length(B), length(C)))
    ((size(A, 1) == length(B)) && (size(A, 2) == length(C))) || throw(DimensionMismatch())
    m, n = size(A)
    for t in C
        if isa(t, MPOTensor)
            isoneunit(space_l(t)) || throw(SpaceMismatch("left space must be vacuum by convention"))
        end
    end
    for t in B
        if isa(t, MPOTensor)
            isoneunit(space_r(t)) || throw(SpaceMismatch("only strict MPOTensor is allowed"))
        end
    end
    data = Matrix{Any}(undef, m+2, n+2)
    data[1, 1] = 1
    data[end, end] = 1
    data[2:end, 1] .= 0
    data[end, 1:end-1] .= 0
    data[1, 2:end-1] = C
    data[1, end] = D
    data[2:end-1, end] = B
    data[2:end-1, 2:end-1] = A
    return SchurMPOTensor(data)
end
