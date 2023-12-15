println("------------------------------------")
println("|         MPOHamiltonian           |")
println("------------------------------------")


@testset "MPOHamiltonian: nearest-neighbour XXZ" begin
	p = spin_site_ops_u1()
	for L in (2,3,4)
		hz = rand(L)
		J = rand(L)
		Jzz = rand(L)
		h1 = nn_mpoham(hz, J, Jzz, p)
		@test space_l(h1) == oneunit(space_l(h1))
		@test space_r(h1)' == oneunit(space_r(h1))
		@test length(h1) == L
		h2 = nn_ham(hz, J, Jzz, p)
		@test length(h2) == L
		@test physical_spaces(h1) == physical_spaces(h2)
		right = iseven(L) ? Rep[U₁](0=>1) : Rep[U₁](1=>1)
		state = randommps(ComplexF64, physical_spaces(h1), right=right, D=4)
		state = canonicalize!(state)
		@test expectation(h1, state) ≈ expectation(h2, state) atol = 1.0e-12
		@test distance(MPO(h1), h2) ≈ 0. atol = 1.0e-5		
	end
end
