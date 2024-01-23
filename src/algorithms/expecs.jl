"""
    expectation(psiA::MPS, h::MPO, psiB::MPS)
    expectation(h::MPO, psi::MPS) = expectation(psi, h, psi)

Return < psiA | h | psiB >
"""
function expectation(psiA::AbstractFiniteMPS, h::MPO, psiB::AbstractFiniteMPS) 
    (length(psiA) == length(h) == length(psiB)) || throw(DimensionMismatch())
    hold = r_RR(psiA, h, psiB)
    for i in length(psiA):-1:1
        hold = updateright(hold, psiA[i], h[i], psiB[i])
    end
    return scalar(hold) 
end
expectation(psiA::AbstractFiniteMPS, h::AdjointMPO, psiB::AbstractFiniteMPS) = conj(expectation(psiB, h.parent, psiA))
expectation(h::MPO, psi::AbstractFiniteMPS) = expectation(psi, h, psi) 
"""
    expectationvalue(h::MPO, psi::AbstractFiniteMPS) 

return expectation(h, psi) / dot(psi, psi)
"""
expectationvalue(h::Union{MPO, MPOHamiltonian}, psi::AbstractFiniteMPS) = expectation(h, psi) / dot(psi, psi)

# partial MPO
function expectation_canonical(m::PartialMPO, psi::MPS)
    ((positions(m)[1] >= 1) && (positions(m)[end] <= length(psi))) || throw(BoundsError())
    return _expectation_canonical(m, psi) 
end

function expectation(psiA::AbstractFiniteMPS, m::PartialMPO, psiB::AbstractFiniteMPS, envs=environments(psiA, psiB))
    cstorage = envs.cstorage
    (length(psiA) == length(psiB) == length(cstorage)-1) || throw(DimensionMismatch())
    isempty(m) && return 0.
    L = length(psiA)
    pos = positions(m)
    ops = storage(m)
    pos_end = pos[end]
    ((pos[1] >= 1) && (pos_end <= L)) || throw(BoundsError())
    util = get_trivial_leg(psiA[1])
    @tensor hold[-3 -2; -1] := conj(psiA[pos_end][-1, 1, 2]) * cstorage[pos_end+1][3, 2] * psiB[pos_end][-3, 5, 3] * ops[end][-2, 1, 4, 5] * util[4]  
    for j in pos_end-1:-1:pos[1]
        pj = findfirst(x->x==j, pos)
        if isnothing(pj)
            hold = updateright(hold, psiA[j], pj, psiB[j])
        else
            hold = updateright(hold, psiA[j], ops[pj], psiB[j])
        end
    end
    @tensor hnew[-2; -1] := conj(util[1]) * hold[-2, 1, -1]
    for j in pos[1]-1:-1:1
        hnew = updateright(hnew, psiA[j], psiB[j])
    end
    return scalar(hnew) 
end

expectation(m::PartialMPO, psi::AbstractFiniteMPS, envs=environments(psi, psi)) = expectation(psi, m, psi, envs) 
expectationvalue(m::PartialMPO, psi::AbstractFiniteMPS, envs=environments(psi, psi)) = expectation(m, psi, envs) / value(envs)

"""
    assume the underlying state is canonical
"""
function _expectation_canonical(m::PartialMPO, psi::AbstractMPS)
    isempty(m) && return 0.
    pos = positions(m)
    ops = storage(m)
    pos_end = pos[end]
    util = get_trivial_leg(psi[1])
    @tensor hold[-3 -2; -1] := conj(psi[pos_end][-1, 1, 2]) * psi[pos_end][-3, 4, 2] * ops[end][-2, 1, 3, 4] * util[3] 
    for j in pos_end-1:-1:pos[1]
        pj = findfirst(x->x==j, pos)
        if isnothing(pj)
            hold = updateright(hold, psi[j], pj, psi[j])
        else
            hold = updateright(hold, psi[j], ops[pj], psi[j])
        end
    end  
    # s = convert(TensorMap, psi.s[pos[1]]) 
    s =  psi.s[pos[1]]
    @tensor hnew[-1; -2] := conj(s[-1, 1]) * hold[3, 2, 1] * conj(util[2]) * s[-2, 3]
    return tr(hnew) 
end

function expectation(psiA::AbstractFiniteMPS, h::MPOHamiltonian, psiB::AbstractFiniteMPS)
    (length(psiA) == length(psiB) == length(h)) || throw(DimensionMismatch())
    hold = r_RR(psiA, h, psiB)
    for i in length(psiA):-1:1
        hold = updateright(hold, psiA[i], h[i], psiB[i])
    end
    hleft = l_LL(psiA, h, psiB)
    (length(hleft) == length(hold)) || error("something wrong")
    T = promote_type(scalartype(psiA), scalartype(h), scalartype(psiB))
    r = zero(T)
    for (a, b) in zip(hleft, hold)
        r += @tensor tmp = a[1,2,3] * b[3,2,1]
    end
    return r 
end
expectation(h::MPOHamiltonian, psi::AbstractFiniteMPS) = expectation(psi, h, psi)


get_trivial_leg(m::AbstractTensorMap) = TensorMap(ones,scalartype(m),oneunit(space(m,1)), one(space(m,1)))
