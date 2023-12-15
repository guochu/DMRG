

struct RescaledMPS{A<:MPSTensor, B<:MPSBondTensor} <: AbstractMPS{A}
	data::Vector{A}
	svectors::Vector{B}
	coeff::Ref{Float64}
end

coeff(x::RescaledMPS) = x.coeff[]
overall_scale(x::RescaledMPS) = coeff(x)^(length(x))
rescaling!(psi::RescaledMPS, n::Real) = _rescaling!(psi, n)
function _rescaling!(psi, n::Real)
	L = length(psi)
	scale1 = n^(1/L)
	psi.coeff[] = coeff(psi) * scale1
	return psi
end
