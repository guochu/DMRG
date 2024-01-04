LinearAlgebra.dot(hA::MPO, hB::MPO) = _dot(hA, hB) 
LinearAlgebra.norm(h::MPO) = sqrt(abs(real(_dot(h, h)))) 
distance2(hA::MPO, hB::MPO) = _distance2(hA, hB)
distance(hA::MPO, hB::MPO) = _distance(hA, hB)


# get identity operator

"""
    TK.id(m::MPO)
Retuen an identity MPO from a given MPO
"""
TK.id(m::MPO) = MPO([id(Matrix{scalartype(m)}, oneunit(spacetype(m)) ⊗ space(item, 2) ) for item in m.data])

TK.id(m::PartialMPO) = PartialMPO([id(Matrix{scalartype(m)}, oneunit(spacetype(m)) ⊗ space(item, 2) ) for item in m.data], positions(m))


l_tr(h::MPO) = TensorMap(ones,scalartype(h),oneunit(spacetype(h))')

function LinearAlgebra.tr(h::MPO)
    isempty(h) && return 0.
    L = length(h)
    hold = l_tr(h)
    for i in 1:L
        hold = updatetraceleft(hold, h[i])
    end
    return scalar(hold) 
end


function LinearAlgebra.lmul!(f::Number, h::Union{MPO, PartialMPO})
    if !isempty(h)
        h[1] *= f
    end
    _renormalize!(h, h[1], false)
    return h
end
LinearAlgebra.rmul!(h::Union{MPO, PartialMPO}, f::Number) = lmul!(f, h)

Base.:*(h::Union{MPO, PartialMPO}, f::Number) = lmul!(f, copy(h))
Base.:*(f::Number, h::Union{MPO, PartialMPO}) = h * f
Base.:/(h::Union{MPO, PartialMPO}, f::Number) = h * (1/f)



"""
    Base.:+(hA::M, hB::M) where {M <: MPO}
    addition of two MPOs
"""
function Base.:+(hA::MPO, hB::MPO)
    @assert !isempty(hA)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    (space_r(hA)==space_r(hB)) || throw(SpaceMismatch())
    T = common_scalartype(hA, hB)
    S = spacetype(hA)
    M = mpotensortype(S, T)
    scale_a = coeff(hA)
    scale_b = coeff(hB)
    if length(hA) == 1
        return MPO([scale_a * hA[1] + scale_b * hB[1]])
    end
    embedders = [left_embedders(T, space_l(hA[i]), space_l(hB[i])) for i in 2:length(hA)]
    r = Vector{M}(undef, length(hA))
    for i in 1:length(hA)
        if i == 1
            @tensor m1[-1 -2; -3 -4] := scale_a * hA[i][-1,-2,1,-4] * (embedders[i][1])'[1, -3]
            @tensor m1[-1 -2; -3 -4] += scale_b * hB[i][-1,-2,1,-4] * (embedders[i][2])'[1, -3]
        elseif i == length(hA)
            @tensor m1[-1 -2; -3 -4] := scale_a * embedders[i-1][1][-1, 1] * hA[i][1,-2,-3,-4] 
            @tensor m1[-1 -2; -3 -4] += scale_b * embedders[i-1][2][-1, 1] * hB[i][1,-2,-3,-4] 
        else          
            @tensor m1[-1 -2; -3 -4] := scale_a * embedders[i-1][1][-1, 1] * hA[i][1,-2,2,-4] * (embedders[i][1])'[2, -3]
            @tensor m1[-1 -2; -3 -4] += scale_b * embedders[i-1][2][-1, 1] * hB[i][1,-2,2,-4] * (embedders[i][2])'[2, -3]
        end
        r[i] = m1 
    end
    return MPO(r)
end

function Base.:+(hA::PartialMPO, hB::PartialMPO)
    @assert !isempty(hA)
    (positions(hA) == positions(hB)) || throw(ArgumentError("only PartialMPOs with same positions are allowed to add"))
    T = promote_type(scalartype(hA), scalartype(hB))
    S = spacetype(hA)
    M = mpotensortype(S, T)
    scale_a = coeff(hA)
    scale_b = coeff(hB)
    if length(hA) == 1
        return PartialMPO([scale_a * hA[1] + scale_b * hB[1]], positions(hA))
    end
    embedders = [left_embedders(T, space_l(hA[i]), space_l(hB[i])) for i in 2:length(hA)]
    r = Vector{M}(undef, length(hA))
    for i in 1:length(hA)
        if i == 1
            @tensor m1[-1 -2; -3 -4] := hA[i][-1,-2,1,-4] * (embedders[i][1])'[1, -3]
            @tensor m1[-1 -2; -3 -4] += scale_b * hB[i][-1,-2,1,-4] * (embedders[i][2])'[1, -3]
        elseif i == length(hA)
            @tensor m1[-1 -2; -3 -4] := scale_a * embedders[i-1][1][-1, 1] * hA[i][1,-2,-3,-4] 
            @tensor m1[-1 -2; -3 -4] += scale_b * embedders[i-1][2][-1, 1] * hB[i][1,-2,-3,-4] 
        else          
            @tensor m1[-1 -2; -3 -4] := scale_a * embedders[i-1][1][-1, 1] * hA[i][1,-2,2,-4] * (embedders[i][1])'[2, -3]
            @tensor m1[-1 -2; -3 -4] += scale_b * embedders[i-1][2][-1, 1] * hB[i][1,-2,2,-4] * (embedders[i][2])'[2, -3]
        end
        r[i] = m1 
    end 
    return PartialMPO(r, positions(hA))       
end

# adding mpo with adjoint mpo will return an normal mpo
Base.:+(hA::MPO, hB::AdjointMPO) = hA + convert(MPO, hB)
Base.:+(hA::AdjointMPO, hB::MPO) = hB + hA
Base.:+(hA::AdjointMPO, hB::AdjointMPO) = adjoint(hA.parent + hB.parent)
Base.:-(hA::AbstractMPO, hB::AbstractMPO) = hA + (-1) * hB
Base.:-(h::AbstractMPO) = -1 * h

"""
    Base.:*(h::MPO, psi::MPS)
    Multiplication of mps by an mpo.
"""
function Base.:*(h::MPO, psi::MPS)
    @assert !isempty(h)
    (length(h) == length(psi)) || throw(DimensionMismatch())
    r = [@tensor tmp[-1 -2; -3 -4 -5] := a[-1, -3, -4, 1] * b[-2, 1, -5] for (a, b) in zip(h.data, psi.data)]
    left = isomorphism(fuse(space_l(h), space_l(psi)), space_l(h) ⊗ space_l(psi))
    fusion_ts = [isomorphism(space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    @tensor tmp[-1 -2; -3] := left[-1, 1, 2] * r[1][1,2,-2,3,4] * fusion_ts[1][3,4,-3]
    mpstensors = Vector{typeof(tmp)}(undef, length(psi))
    mpstensors[1] = tmp
    for i in 2:length(psi)
        @tensor tmp[-1 -2; -3] := conj(fusion_ts[i-1][1,2,-1]) * r[i][1,2,-2,3,4] * fusion_ts[i][3,4,-3]
        mpstensors[i] = tmp
    end
    return MPS(mpstensors)

end
Base.:*(h::MPO, psi::ExactMPS) = ExactMPS(h * MPS(psi))

Base.:*(h::PartialMPO, psi::MPS) = apply!(h, copy(psi))
function apply!(h::PartialMPO, psi::MPS)
    @assert positions(h)[end] <= length(psi)
    M = tensormaptype(spacetype(psi), 2, 3, promote_type(scalartype(h), scalartype(psi)))
    _start, _end = positions(h)[1], positions(h)[end]
    r = Vector{M}(undef, _end - _start + 1)
    leftspace = oneunit(space_l(h))

    for (i, pos) in enumerate(_start:_end)
        if pos in positions(h)
            r[i] = @tensor tmp[-1 -2; -3 -4 -5] := h[i][-1, -3, -4, 1] * psi[pos][-2, 1, -5]
            leftspace = space_r(h[i])'
        else
            hj = id(storagetype(M), leftspace)
            r[i] = @tensor tmp[-1 -2; -3 -4 -5] := hj[-1, -4] * psi[pos][-2, -3, -5]
        end
    end
    fusion_ts = [isomorphism(space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    left = isomorphism(fuse(space(r[1], 1), space(r[1], 2)), space(r[1], 1) ⊗ space(r[1], 2))
    psi[_start] = @tensor tmp[1,4;7] := left[1,2,3] * r[1][2,3,4,5,6] * fusion_ts[1][5,6,7]
    for (i, pos) in enumerate(_start+1:_end)
        psi[pos] = @tensor tmp[3,4;7] := conj(fusion_ts[i][1,2,3]) * r[i+1][1,2,4,5,6] * fusion_ts[i+1][5,6,7]
    end
    return psi
end



"""
    Base.:*(hA::M, hB::M) where {M <: MPO}
    a * b
"""
function Base.:*(hA::MPO, hB::MPO) 
    @assert !isempty(hA)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    return MPO(_mult_n_n(hA.data, hB.data))
end



# the last one of mpotensors is still missing
function _mult_util(a::Vector{M}, b::Vector{<:MPOTensor}) where {M<:MPOTensor}
    r = [@tensor tmp[-1 -2 -3; -4 -5 -6] := aj[-1, -3, -4, 1] * bj[-2, 1, -5, -6] for (aj, bj) in zip(a, b)]
    T = scalartype(r[1])
    # left = isomorphism(Matrix{T}, fuse(space(a[1], 1), space(b[1], 1)), space(a[1], 1) ⊗ space(b[1], 1))
    # fusion_ts = [isomorphism(Matrix{T}, space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r]
    fusion_ts = [isomorphism(Matrix{T}, fuse(space(item, 1), space(item, 2)), space(item, 1) ⊗ space(item, 2)) for item in r]

    mpotensors = Vector{mpotensortype(spacetype(M), scalartype(M))}(undef, length(a))
    if length(a) > 1
        @tensor tmp[-1 -2; -3 -4] := fusion_ts[1][-1, 1, 2] * r[1][1,2,-2,3,4,-4] * conj(fusion_ts[2][-3, 3,4])
        mpotensors[1] = tmp
        for i in 2:length(a)-1
            @tensor tmp[-1 -2; -3 -4] := fusion_ts[i][-1,1,2] * r[i][1,2,-2,3,4,-4] * conj(fusion_ts[i+1][-3,3,4])
            mpotensors[i] = tmp
        end   
    end
    
    return mpotensors, r[end], fusion_ts[end]
end

_right_fusion_tensor_n_n(ab) = isomorphism(Matrix{scalartype(ab)}, space(ab, 4)' ⊗ space(ab, 5)', fuse(space(ab, 4)', space(ab, 5)'))
function _right_fusion_tensor_a_n(ab)
     if isoneunit(space(ab, 4)) || isoneunit(space(ab, 5))
        right = isomorphism(Matrix{scalartype(ab)}, space(ab, 4)' ⊗ space(ab, 5)', fuse(space(ab, 4)', space(ab, 5)') )
    else
        right = isomorphism(Matrix{scalartype(ab)}, space(ab, 4)', space(ab, 5) ⊗ oneunit(space(ab, 5)) )
    end   
end

function _mult_finish(mpotensors, ab, fusion_ts, right)
    @tensor tmp[-1 -2; -3 -4] := fusion_ts[-1, 1, 2] * ab[1,2,-2,3,4,-4] * right[3,4,-3]
    mpotensors[end] = tmp
    return mpotensors
end


function _mult_n_n(a::Vector{<:MPOTensor}, b::Vector{<:MPOTensor})
    mpotensors, ab, fusion_ts = _mult_util(a, b)
    # compute right fusion tensor
    right = _right_fusion_tensor_n_n(ab)
    return _mult_finish(mpotensors, ab, fusion_ts, right)
end

"""
    Base.:*(hA::M, hB::M) where {M <: AdjointMPO} 
    a† * b†
"""
Base.:*(hA::AdjointMPO, hB::AdjointMPO) = (hB.parent * hA.parent)'


"""
    Base.:*(hA::AdjointMPO, hB::MPO)
    a† * b
"""
function Base.:*(hA::AdjointMPO, hB::MPO) 
    @assert !isempty(hB)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    check_allowed_fusion(space_r(hA.parent)', space_r(hB))
    return MPO(_mult_a_n(hA.parent.data,  hB.data))
end

function _mult_a_n(a::Vector{<:MPOTensor}, b::Vector{<:MPOTensor})
    a = mpotensor_adjoint.(a)
    mpotensors, ab, fusion_ts = _mult_util(a, b)

    right = _right_fusion_tensor_a_n(ab)
    return _mult_finish(mpotensors, ab, fusion_ts, right)
end

"""
    Base.:*(hA::MPO, hB::AdjointMPO)
    a * b†
"""
function Base.:*(hA::MPO, hB::AdjointMPO)
    @assert !isempty(hA)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    check_allowed_fusion(space_r(hA), space_r(hB.parent)')
    return MPO(_mult_n_a(hA.data, hB.parent.data))
end

function _mult_n_a(a::Vector{<:MPOTensor}, b::Vector{<:MPOTensor})
    b = mpotensor_adjoint.(b)
    mpotensors, ab, fusion_ts = _mult_util(a, b)

    right = _right_fusion_tensor_a_n(ab)
    return _mult_finish(mpotensors, ab, fusion_ts, right)
end



const MPO_APPROX_EQUAL_ATOL = 1.0e-12

"""
    Base.isapprox(a::M, b::M) where {M <: MPO} 
    Check is two MPOs are approximated equal 
"""
Base.isapprox(a::MPO, b::MPO; atol=MPO_APPROX_EQUAL_ATOL) = distance2(a, b) <= atol

"""
    r_RR(psiA::AbstractMPS, h::MPO, psiB::AbstractMPS)

Notice the convention!!!
a is bra, b is ket, ^
                    a
                    -
                    h
                    -
                    b
                    v
for r_RR b is codomain, namely 
r_RR =  ----3
        ----2
        ----1
for l_LL a is codomain, namely
        ----1
        ----2
l_LL =  ----3
"""
r_RR(psiA::AbstractMPS, h::MPO, psiB::AbstractMPS) = loose_isometry(common_storagetype(psiA, h, psiB), space_r(psiB)' ⊗ space_r(h)', space_r(psiA)')
l_LL(psiA::AbstractMPS, h::MPO, psiB::AbstractMPS) = loose_isometry(common_storagetype(psiA, h, psiB), space_l(psiA) ⊗ space_l(h)', space_l(psiB))


"""
    expectation(psiA::MPS, h::MPO, psiB::MPS)
    expectation(h::MPO, psi::MPS) = expectation(psi, h, psi)

Return < psiA | h | psiB >
"""
function expectation(psiA::AbstractMPS, h::MPO, psiB::AbstractMPS) 
    (length(psiA) == length(h) == length(psiB)) || throw(DimensionMismatch())
    hold = r_RR(psiA, h, psiB)
    for i in length(psiA):-1:1
        hold = updateright(hold, psiA[i], h[i], psiB[i])
    end
    return scalar(hold) 
end
expectation(psiA::AbstractMPS, h::AdjointMPO, psiB::AbstractMPS) = conj(expectation(psiB, h.parent, psiA))
expectation(h::MPO, psi::AbstractMPS) = expectation(psi, h, psi)

function LinearAlgebra.ishermitian(h::MPO)
    isempty(h) && throw(ArgumentError("input operator is empty."))
    isstrict(h) || return false
    return isapprox(h, convert(MPO, h'), atol=1.0e-10) 
end

# only for these two cases, the fusion of AdjointMPO and MPO can be unambiguously determined
function check_allowed_fusion(x::S, y::S) where {S <: ElementarySpace}
    if isdual(x) != isdual(y)
        ((x == y') || (isoneunit(x) || isoneunit(y)) ) || throw(ArgumentError("Only two modes are supported: Either x and y are dual or one is vaccum"))
    end
end
