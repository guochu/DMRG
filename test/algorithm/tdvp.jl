println("------------------------------------")
println("----------|    TDVP    |------------")
println("------------------------------------")


function do_tdvp(dmrg, alg, n, obs)
	for i in 1:n
		sweep!(dmrg, alg)
	end
	return real([expectation(item, dmrg.state) for item in obs])
end

@testset "TDVP: comparison with exact evolution" begin
	J = 1.
	J2 = 1.2
	U = 0.7

	dt = 0.01
	dmrg_sweeps = 50
	tol = 1.0e-4
	for L in (3, 4)


		mpo, observers = hubbard_ladder(L, J, J2, U, spinal_fermion_site_ops_u1_u1())
		state = initial_state_u1_u1(ComplexF64, L)
		canonicalize!(state)

		state1 = exact_timeevolution(mpo, -im*dt*dmrg_sweeps, ExactMPS(state), ishermitian=true)
		obs1 = real([expectation(item, state1) for item in observers])

		obs2 = do_tdvp(environments(mpo, copy(state)), TDVP2(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers)
		@test max_error(obs1, obs2) < tol

		obs3 = do_tdvp(environments(mpo, copy(state)), TDVP1S(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers)
		@test max_error(obs1, obs3) < tol

		mpo, observers = hubbard_ladder(L, J, J2, U, spinal_fermion_site_ops_u1_su2())

		state = initial_state_u1_su2(ComplexF64, L)
		canonicalize!(state)

		state1 = exact_timeevolution(mpo, -im*dt*dmrg_sweeps, ExactMPS(state), ishermitian=true)
		obs4 = real([expectation(item, state1) for item in observers])
		@test max_error(obs1, obs4) < 1.0e-8

		obs5 = do_tdvp(environments(mpo, copy(state)), TDVP2(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers)
		@test max_error(obs1, obs5) < tol

		obs6 = do_tdvp(environments(mpo, copy(state)), TDVP1S(stepsize=-dt*im, ishermitian=true), dmrg_sweeps, observers)
		@test max_error(obs1, obs6) < tol

	end

end