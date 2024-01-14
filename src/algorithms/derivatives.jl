
function ac_prime(x::MPSTensor, m::MPOTensor, hleft::MPSTensor, hright::MPSTensor) 
	@tensor tmp[-1 -2; -3] := ((hleft[-1, 1, 2] * x[2,3,4]) * m[1,-2,5,3]) * hright[4,5,-3]
end


function ac2_prime(x::MPOTensor, h1::MPOTensor, h2::MPOTensor, hleft::MPSTensor, hright::MPSTensor)
	@tensor tmp[-1 -2; -3 -4] := (((hleft[-1, 1, 2] * x[2, 3, 4, 5]) * h1[1, -2, 6, 3]) * h2[6, -3, 7, 4]) * hright[5, 7, -4]
end

function ac_prime(x::MPSTensor{S}, ham::AbstractSparseMPOTensor{S}, hleft::Vector, hright::Vector) where {S <: ElementarySpace}
    tmp = zero(x)
    for (i,j) in opkeys(ham)
        @tensor tmp[-1,-2,-3]+=hleft[i][-1,5,4]*x[4,2,1]*ham[i,j][5,-2,3,2]*hright[j][1,3,-3]
    end
    for (i,j) in scalkeys(ham)
        scal = ham.Os[i,j]
        @tensor tmp[-1,-2,-3]+=hleft[i][-1,5,4]*(scal*x)[4,-2,1]*hright[j][1,5,-3]
    end
    return tmp
end

function ac2_prime(x::MPOTensor{S},h1::AbstractSparseMPOTensor{S},h2::AbstractSparseMPOTensor{S},hleft::Vector, hright::Vector) where {S <: ElementarySpace}
	@assert size(h1, 2) == size(h2, 1)
    tmp=zero(x)

    for (i,j) in keys(h1)
        for k in 1:size(h2, 2)
            contains(h2,j,k) || continue

            if isscal(h1,i,j) && isscal(h2,j,k)
                scal = h1.Os[i,j]*h2.Os[j,k]
                @tensor tmp[-1,-2,-3,-4] += (scal*hleft[i])[-1,7,6]*x[6,-2,-3,1]*hright[k][1,7,-4]
            elseif isscal(h1,i,j)
                scal = h1.Os[i,j]
                @tensor tmp[-1,-2,-3,-4]+=(scal*hleft[i])[-1,7,6]*x[6,-2,3,1]*h2[j,k][7,-3,2,3]*hright[k][1,2,-4]
            elseif isscal(h2,j,k)
                scal = h2.Os[j,k]
                @tensor tmp[-1,-2,-3,-4]+=(scal*hleft[i])[-1,7,6]*x[6,5,-3,1]*h1[i,j][7,-2,2,5]*hright[k][1,2,-4]
            else
                @tensor tmp[-1,-2,-3,-4]+=hleft[i][-1,7,6]*x[6,5,3,1]*h1[i,j][7,-2,4,5]*h2[j,k][4,-3,2,3]*hright[k][1,2,-4]
            end
        end

    end

    return tmp
end


function c_prime(x::MPSBondTensor, hleft::MPSTensor, hright::MPSTensor) 
    @tensor tmp[-1; -2] := (hleft[-1, 1, 2] * x[2, 3]) * hright[3, 1, -2]
end

function c_prime(x::MPSBondTensor, hleft::Vector, hright::Vector)
    @assert length(hleft) == length(hright)
    tmp = zero(x)
    for (hl, hr) in zip(hleft, hright)
        @tensor tmp[-1, -2] += (hl[-1, 1, 2] * x[2, 3]) * hr[3,1,-2]
    end
    return tmp
end



# struct OverlapAprime{L<:MPSBondTensor, R<:MPSBondTensor}
#     left::L
#     right::R
# end

# ∂A(left::MPSBondTensor, right::MPSBondTensor) = OverlapAprime(left, right)

# function Base.:*(m::OverlapAprime, v::MPSTensor)
#     @tensor tmp[1,3;5] := m.left[1,2] * v[2,3,4] * m.right[4,5]
# end

# struct ExpecAprime{M<:Union{MPOTensor, AbstractSparseMPOTensor}, L<:Union{MPSTensor, Vector}, R<:Union{MPSTensor, Vector}}
#     left::L
#     m::M
#     right::R
# end
# ∂A(left::MPSTensor, m::MPOTensor, right::MPSTensor) = ExpecAprime(left, m, right)

# function Base.:*(m::ExpecAprime{<:MPOTensor}, v::MPSTensor)
#     # @tensor tmp[1,6;8] := m.left[1,2,3] * v[3,4,5] * m.m[2,6,7,4] * m.right[5,7,8]
#     return ac_prime(v, m.m, m.left, m.right)
# end
# function Base.:*(m::ExpecAprime{<:AbstractSparseMPOTensor}, v::MPSTensor)
#     return ac_prime(v, m.m, m.left, m.right)
# end
