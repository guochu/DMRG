
struct RescaledMPO{A <: MPOTensor} <: AbstractMPO{A}
	data::Vector{A}
	coeff::Ref{Float64}
end