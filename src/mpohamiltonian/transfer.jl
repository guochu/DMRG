

function r_RR(psiA::M, h::MPOHamiltonian, psiB::M) where {M <: Union{MPS, ExactMPS}}
	(length(psiA) == length(psiB) == length(h)) || throw(DimensionMismatch())
	L = length(psiA)
	T = common_scalartype(psiA, h, psiB)
	rrr = r_RR(psiA)

	i = size(h[L], 2)
	util_right = TensorMap(ones,T,h[L].rightspaces[i], one(h[L].rightspaces[i]))
	@tensor ctr[-1 -2; -3] := rrr[-1,-3]*util_right[-2]

	right_starter = Vector{typeof(ctr)}(undef, size(h[L], 2))
	right_starter[end] = ctr
	for i in 1:size(h[L], 2)-1
		util_right = TensorMap(zeros,T,h[L].rightspaces[i], one(h[L].rightspaces[i]))

		@tensor ctr[-1 -2; -3]:= rrr[-1,-3]*util_right[-2]
		right_starter[i] = ctr
	end
	return right_starter
end 


function l_LL(psiA::M, h::MPOHamiltonian, psiB::M) where {M <: Union{MPS, ExactMPS}}
	(length(psiA) == length(psiB) == length(h)) || throw(DimensionMismatch())
	L = length(psiA)
	T = common_scalartype(psiA, h, psiB)
	lll = l_LL(psiA)

	i = 1
	util_left = TensorMap(ones,T,h[1].leftspaces[i]', one(h[1].leftspaces[i]))
	@tensor ctl[-1 -2; -3]:= lll[-1,-3]*util_left[-2]
	left_starter = [ctl]

	for i in 2:size(h[1], 1)
		util_left = TensorMap(zeros,T,h[1].leftspaces[i]', one(h[1].leftspaces[i]))
		@tensor ctl[-1 -2; -3]:= lll[-1,-3]*util_left[-2]
		push!(left_starter, ctl)
	end
	return left_starter
end 



function updateright(hold::Vector, psiAj::MPSTensor{S}, hj::AbstractSparseMPOTensor{S}, psiBj::MPSTensor) where {S <: ElementarySpace}
	@assert length(hold) == size(hj, 2)
	T = common_scalartype(psiAj, hj, psiBj)
	# hnew = [TensorMap(zeros, T, space(psiAj, 1)' , hj.leftspaces[i]' ⊗ space(psiBj, 1)' ) for i in 1:size(hj, 1) ]
	hnew = [TensorMap(zeros, T, space(psiBj, 1) ⊗ hj.leftspaces[i], space(psiAj, 1) ) for i in 1:size(hj, 1) ]
	for (i, j) in keys(hj)
		if isscal(hj, i, j)
			hnew[i] += hj.Os[i, j] * updateright(hold[j], psiAj, nothing, psiBj)
		else
			hnew[i] += updateright(hold[j], psiAj, hj.Os[i, j], psiBj)
		end
	end
	return hnew
end

function updateleft(hold::Vector, psiAj::MPSTensor{S}, hj::AbstractSparseMPOTensor{S}, psiBj::MPSTensor) where {S <: ElementarySpace}
	@assert length(hold) == size(hj, 1)
	T = common_scalartype(psiAj, hj, psiBj)
	hnew = [TensorMap(zeros, T, space(psiAj, 3)' ⊗ hj.rightspaces[j]' , space(psiBj, 3)' ) for j in 1:size(hj, 2)]

	for (i, j) in keys(hj)
		if isscal(hj, i, j)
			hnew[j] += hj.Os[i, j] * updateleft(hold[i], psiAj, nothing, psiBj)
		else
			hnew[j] += updateleft(hold[i], psiAj, hj.Os[i, j], psiBj)
		end
	end
	return hnew
end


