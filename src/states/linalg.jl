LinearAlgebra.dot(psiA::AbstractFiniteMPS, psiB::AbstractFiniteMPS) = _dot(psiA, psiB) 
function _dot(a, b)
    (length(a) == length(b)) || throw(DimensionMismatch())
    hold = l_LL(a, b)
    for i in 1:length(a)
        hold = updateleft(hold, a[i], b[i])
    end
    return tr(hold)    
end

function LinearAlgebra.norm(psi::AbstractFiniteMPS; iscanonical::Bool=false) 
	n = iscanonical ? norm(psi[1]) : sqrt(abs(real(_dot(psi, psi))))
    return n 
end
LinearAlgebra.norm(psi::ExactMPS) = norm(psi[psi.center])

"""
    distance(a, b)
Square of Euclidean distance between a and b.
"""
distance2(a::AbstractMPS, b::AbstractMPS) = _distance2(a, b)

"""
    distance(a, b)
Euclidean distance between a and b.
"""
distance(a::AbstractMPS, b::AbstractMPS) = _distance(a, b)

function ac_prime(x::MPSTensor, cleft::MPSBondTensor, cright::MPSBondTensor) 
    @tensor tmp[-1 -2; -3] := cleft[-1, 1] * x[1, -2, 2] * cright[2, -3]
end

function LinearAlgebra.normalize!(psi::MPS; iscanonical::Bool=false)
    if !iscanonical
        canonicalize!(psi, alg=Orthogonalize(QR(), normalize=true))
    else
        normalize!(psi[1])
        _renormalize_coeff!(psi, true)
    end
    return psi
end
function LinearAlgebra.normalize!(psi::ExactMPS)
    normalize!(psi[psi.center])
    return psi
end
LinearAlgebra.normalize(psi::AbstractFiniteMPS; kwargs...) = normalize!(copy(psi); kwargs...)
function LinearAlgebra.lmul!(f::Number, psi::MPS)
    if !isempty(psi)
        psi[1] *= f
    end
    _renormalize!(psi, psi[1], false)
    return psi
end
LinearAlgebra.rmul!(psi::MPS, f::Number) = lmul!(f, psi)

Base.:*(psi::MPS, f::Number) = lmul!(f, copy(psi))
Base.:*(f::Number, psi::MPS) = psi * f
Base.:/(psi::MPS, f::Number) = psi * (1/f)


# Von neuman entropy

"""
    renyi_entropy(psi::MPS; bond, α)
"""
function renyi_entropy(psi::MPS; bond::Int=div(length(psi)+1, 2), kwargs...)
    r = entanglement_spectrum(psi.s[bond+1])
    for i in 1:length(r)
        r[i] = r[i]^2
    end
    return renyi_entropy(r; kwargs...)
end
renyi_entropies(psi::MPS; kwargs...) = [renyi_entropy(psi; bond=bond, kwargs...) for bond in 1:length(psi)-1]


"""
    Base.:+(psiA::M, psiB::M) where {M <: MPS}
Addition of two MPSs
"""
function Base.:+(psiA::MPS, psiB::MPS) 
    (length(psiA) == length(psiB)) || throw(DimensionMismatch())
    @assert !isempty(psiA)
    (space_r(psiA)==space_r(psiB)) || throw(SpaceMismatch())
    scale_a = scale_b = one(scalartype(psiA))
    (length(psiA) == 1) && return MPS([scale_a * psiA[1] + scale_b * psiB[1]])

    T = promote_type(scalartype(psiA), scalartype(psiB))
    embedders = [right_embedders(T, space_r(aj)', space_r(bj)') for (aj, bj) in zip(psiA.data, psiB.data)]
    A = mpstensortype(spacetype(psiA), T)
    r = A[]
    for i in 1:length(psiA)
        if i == 1
            @tensor m1[-1 -2; -3] := scale_a * psiA[i][-1,-2,2] * embedders[i][1][2, -3]
            @tensor m1[-1 -2; -3] += scale_b * psiB[i][-1,-2,2] * embedders[i][2][2, -3]
        elseif i == length(psiA)
            @tensor m1[-1 -2; -3] := scale_a * (embedders[i-1][1])'[-1, 1] * psiA[i][1,-2,-3] 
            @tensor m1[-1 -2; -3] += scale_b * (embedders[i-1][2])'[-1, 1] * psiB[i][1,-2,-3] 
        else          
            @tensor m1[-1 -2; -3] := scale_a * (embedders[i-1][1])'[-1, 1] * psiA[i][1,-2,2] * embedders[i][1][2, -3]
            @tensor m1[-1 -2; -3] += scale_b * (embedders[i-1][2])'[-1, 1] * psiB[i][1,-2,2] * embedders[i][2][2, -3]
        end
        push!(r, m1)
    end
    return MPS(r)
end
Base.:-(psiA::MPS, psiB::MPS) = psiA + (-1) * psiB
Base.:-(psi::MPS) = -1 * psi

const MPS_APPROX_EQUAL_ATOL = 1.0e-14
Base.isapprox(psiA::AbstractMPS, psiB::AbstractMPS; atol=MPS_APPROX_EQUAL_ATOL) = distance2(psiA, psiB) <= atol

