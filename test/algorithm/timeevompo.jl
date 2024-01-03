println("------------------------------------")
println("--------|    TimeEvoMPO    |--------")
println("------------------------------------")


@testset "TimeEvoMPO, real time: comparison with exact evolution" begin
	p = spin_site_ops_u1()

	dt = 0.01
	dmrg_sweeps = 20
	tol = 1.0e-2
	for L in (3, 4)
		# println("test for L= ", L)

		hz = rand(L)
		J = rand(L)
		Jzz = rand(L)
		h = nn_mpoham(hz, J, Jzz, p)
		state = randommps(ComplexF64, physical_spaces(h), D=4, right=Rep[U₁](div(L, 2)=>1))
		observers = [prodmpo(L, [i], [p["z"]]) for i in 1:L]

		state1 = exact_timeevolution(h, -im*dt*dmrg_sweeps, ExactMPS(state), ishermitian=true)
		obs1 = real([expectation(item, state1) for item in observers])

		U = MPO(timeevompo(h, -im*dt, alg=WI()))
		state2 = state
		for i in 1:dmrg_sweeps
			state2 = U * state2
			canonicalize!(state2, alg=Orthogonalize(trunc=truncdimcutoff(D=100, ϵ=1.0e-8)))
		end
		obs2 = real([expectation(item, state2) for item in observers])

		# println(max_error(obs1, obs2))
		@test max_error(obs1, obs2) < tol

		U = MPO(timeevompo(h, -im*dt, alg=WII()))
		state2 = state
		for i in 1:dmrg_sweeps
			state2 = U * state2
			canonicalize!(state2, alg=Orthogonalize(trunc=truncdimcutoff(D=100, ϵ=1.0e-8)))
		end
		obs3 = real([expectation(item, state2) for item in observers])

		# println(max_error(obs1, obs3))
		@test max_error(obs1, obs3) < tol
	end
end

@testset "TimeEvoMPO, imaginary time: comparison with exact evolution" begin
	p = spin_site_ops_u1()

	dmrg_sweeps = 20
	tol = 1.0e-2
	for L in (3, 4)
		for dt in (0.01, -0.01)

			hz = rand(L)
			J = rand(L)
			Jzz = rand(L)
			h = nn_mpoham(hz, J, Jzz, p)
			state = randommps(ComplexF64, physical_spaces(h), D=4, right=Rep[U₁](div(L, 2)=>1))
			observers = [prodmpo(L, [i], [p["z"]]) for i in 1:L]

			state1 = exact_timeevolution(h, -dt*dmrg_sweeps, ExactMPS(state), ishermitian=true)
			obs1 = [expectation(item, state1) / dot(state1, state1) for item in observers]

			U = MPO(timeevompo(h, -dt, alg=WI()))
			state2 = state
			for i in 1:dmrg_sweeps
				state2 = U * state2
				canonicalize!(state2, alg=Orthogonalize(trunc=truncdimcutoff(D=100, ϵ=1.0e-8)))
			end
			obs2 = [expectation(item, state2) / dot(state2, state2) for item in observers]

			# println(max_error(obs1, obs2))
			@test max_error(obs1, obs2) < tol

			U = MPO(timeevompo(h, -dt, alg=WII()))
			state2 = state
			for i in 1:dmrg_sweeps
				state2 = U * state2
				canonicalize!(state2, alg=Orthogonalize(trunc=truncdimcutoff(D=100, ϵ=1.0e-8)))
			end
			obs3 = [expectation(item, state2) / dot(state2, state2) for item in observers]

			# println(max_error(obs1, obs3))
			@test max_error(obs1, obs3) < tol	

		end

	end

end