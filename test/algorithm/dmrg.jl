println("------------------------------------")
println("----------|    DMRG    |------------")
println("------------------------------------")


function do_dmrg(dmrg, alg)
	dmrg_sweeps = 5
	# Evals, delta = compute!(dmrg, alg)
	Evals = Float64[]
	for i in 1:dmrg_sweeps
		Evals, delta = sweep!(dmrg, alg)
	end
	return Evals[end]
end

@testset "DMRG with MPO: comparison with ED" begin
	J = 1.
	J2 = 1.2
	U = 1.37
	for L in 4:5
		for p in (spinal_fermion_site_ops_u1_u1(), spinal_fermion_site_ops_u1_su2())

			mpo, observers = hubbard_ladder(L, J, J2, U, p)

			if spacetype(mpo) == Rep[U₁×U₁]
				state = initial_state_u1_u1(Float64, L)
			else
				state = initial_state_u1_su2(Float64, L)
			end	
			state = randommps(scalartype(state), physical_spaces(state), right=space_r(state)', D=10)
			state = canonicalize!(state, alg=Orthogonalize(normalize=true))

			# ED energy
			E, _st = exact_diagonalization(mpo, right=space_r(state)', num=1, ishermitian=true)
			E = E[1]

			E1 = do_dmrg(environments(mpo, copy(state)), DMRG2())
			@test E ≈ E1 atol = 1.0e-6

			E2 = do_dmrg(environments(mpo, copy(state)), DMRG1S())
			@test E ≈ E2 atol = 1.0e-6

			# check excited state
			dmrg = environments(mpo, copy(state))
			do_dmrg(dmrg, DMRG2())
			gs_state = dmrg.state

			E3, _st = exact_diagonalization(mpo, right=space_r(state)', num=2, ishermitian=true)
			@test E3[1] ≈ E atol = 1.0e-12
			E3 = E3[2]

			E4 = do_dmrg(environments(mpo, copy(state), [gs_state]), DMRG2(trunc=truncdimcutoff(D=20, ϵ=1.0e-10)))
			@test E3 ≈ E4 atol = 1.0e-6

			E5 = do_dmrg(environments(mpo, copy(state), [gs_state]), DMRG1S(trunc=truncdimcutoff(D=20, ϵ=1.0e-10)))
			@test E3 ≈ E5 atol = 1.0e-6

		end
	end
end



