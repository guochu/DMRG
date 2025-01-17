

"""
	updateright(hold::MPSBondTensor, mpsAj::MPSTensor{S}, mpsBj::MPSTensor{S}) where {S<:ElementarySpace}
	update storage from right to left for overlap of mps
"""
function updateright(hold::MPSBondTensor, mpsAj::MPSTensor, mpsBj::MPSTensor) 
	@tensor hnew[1;5] := mpsBj[1,2,3] * hold[3,4] * conj(mpsAj[5,2,4])
end

"""
	updateleft(hold::MPSBondTensor, mpsAj::MPSTensor{S}, mpsBj::MPSTensor{S}) where {S<:ElementarySpace}
	update storage from left to right for overlap of mps
"""
function updateleft(hold::MPSBondTensor, mpsAj::MPSTensor, mpsBj::MPSTensor) 
	@tensor m2[-1 -2; -3] := conj(mpsAj[1, -2, -3]) * hold[1, -1]
	@tensor hnew[-1; -2] := m2[1,2,-1] * mpsBj[1,2,-2]
end


"""
	updateright(hold::MPSBondTensor, hAj::MPOTensor{S}, hBj::MPOTensor{S}) where {S<:ElementarySpace}
	update storage from right to left for overlap of mps
"""
function updateright(hold::MPSBondTensor, hAj::MPOTensor, hBj::MPOTensor) 
	@tensor hnew[1;6] := hBj[1,2,3,4] * hold[3,5] * conj(hAj[6,4,5,2])
end



"""
	updateleft(hold::MPSBondTensor, hAj::MPOTensor{S}, hBj::MPOTensor{S}) where {S<:ElementarySpace}
	update storage from left to right for overlap of mps
"""
function updateleft(hold::MPSBondTensor, hAj::MPOTensor, hBj::MPOTensor) 
	@tensor m2[-1 -2 ; -3 -4] := conj(hAj[1, -2, -3, -4]) * hold[1, -1]
	@tensor hnew[-1; -2] := m2[1,2,-1,3] * hBj[1,2,-2,3]
end

function updateright(hold::MPSTensor, psiAj::MPSTensor, hj::MPOTensor, psiBj::MPSTensor)
	@tensor hnew[1,6;8] := psiBj[1,2,3] * hold[3,4,5] * hj[6,7,4,2] * conj(psiAj[8,7,5])
end

function updateleft(hold::MPSTensor, psiAj::MPSTensor, hj::MPOTensor, psiBj::MPSTensor) 
	@tensor hnew[-1 -2; -3] := hold[1,2,3] * psiBj[3,4,-3] * hj[2,5,-2,4] * conj(psiAj[1,5,-1])
end


function updateright(hold::MPSTensor, psiAj::MPSTensor, hj::Nothing, psiBj::MPSTensor)
	@tensor hnew[1,4;6] := psiBj[1,2,3] * hold[3,4,5] * conj(psiAj[6,2,5])
end

function updateleft(hold::MPSTensor, psiAj::MPSTensor, hj::Nothing, psiBj::MPSTensor) 
	@tensor hnew[-1 -2; -3] := hold[1,-2,2] * psiBj[2,3,-3] * conj(psiAj[1,3,-1])
end

function updateright(hold::MPSTensor, psiAj::MPSTensor, hj::Number, psiBj::MPSTensor)
	@tensor hnew[1,4;6] := hj * psiBj[1,2,3] * hold[3,4,5] * conj(psiAj[6,2,5])
end

function updateleft(hold::MPSTensor, psiAj::MPSTensor, hj::Number, psiBj::MPSTensor) 
	@tensor hnew[-1 -2; -3] := hj * hold[1,-2,2] * psiBj[2,3,-3] * conj(psiAj[1,3,-1])
end

function updatetraceleft(hold::AbstractTensorMap{<:Number, S, 1, 0}, hj::MPOTensor{S}) where {S <: ElementarySpace}
	@tensor hnew[-1] := hold[1] * hj[1,2,-1,2]
	return hnew
end

