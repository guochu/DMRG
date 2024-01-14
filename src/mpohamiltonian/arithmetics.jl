



function expectation(psiA::AbstractMPS, h::MPOHamiltonian, psiB::AbstractMPS)
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
expectation(h::MPOHamiltonian, psi::AbstractMPS) = expectation(psi, h, psi)
