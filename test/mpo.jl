println("------------------------------------")
println("|               MPO                |")
println("------------------------------------")

@testset "MPO initializer: product operator" begin
	# u1 symmetry
	p = spin_site_ops_u1()
	sp, sm, z = p["+"], p["-"], p["z"]
	ph = space(z, 1)
	physpaces = [ph for i in 1:4]
	h1 = prodmpo(Float64, physpaces, [2], [sp])
	@test length(h1) == 4
	@test scalartype(h1) == Float64
	@test scalartype(complex(h1)) == ComplexF64
	for i in 1:length(h1)
		@test !isdual(space(h1[i], 1))
		@test !isdual(space(h1[i], 2))
		@test isdual(space(h1[i], 3))
		@test isdual(space(h1[i], 4))
	end	
	@test bond_dimension(h1) == 1
	@test physical_spaces(h1) == physpaces
	@test space_l(h1) == oneunit(space_l(h1))
	@test space_r(h1) == space_r(sp)
	@test distance(h1, h1) ≈ 0.
	@test norm(2 * h1) ≈ 2 * norm(h1)
	@test norm(h1 / 2) ≈ norm(h1) / 2


	h = h1 * h1'
	@test space_l(h) == oneunit(space_l(h))
	@test space_r(h)' == oneunit(space_r(h))
	@test scalartype(h) == Float64
	@test scalartype(complex(h1) * h1') == ComplexF64
	@test bond_dimension(h) == 1
	h′ = h1' * h1
	@test isa(h′, MPO)
	@test space_l(h′) == oneunit(space_l(h′))
	@test space_r(h′)' == oneunit(space_r(h′))

	h1 = prodmpo(Float64, physpaces, [1, 3], [sp, sp'])
	@test space_l(h1) == oneunit(space_l(h1))
	@test space_r(h1)' == oneunit(space_r(h1))
	@test bond_dimension(h1) == 1
	h2 = convert(MPO, h1')
	@test isa(h2, MPO)
	@test space_l(h2) == oneunit(space_l(h2))
	@test space_r(h2)' == oneunit(space_r(h2))
	h = h1 + h2
	@test bond_dimension(h) == 2
	@test distance(h, convert(MPO, h')) ≈ 0. atol = 1.0e-12

	# su2 symmetry
	p1 = spinal_fermion_site_ops_u1_su2()
	adag_cs, a_cs, adagadag_cs, adaga_cs, nn_cs, n_cs, JW_cs = p1["+"], p1["-"], p1["++"], p1["+-"], p1["n↑n↓"], p1["n"], p1["JW"]
	@tensor adagJW_cs[1,2;3,5] := adag_cs[1,2,3,4] * JW_cs[4,5]
	ops_cs = (adag_cs, adagJW_cs, a_cs, adagadag_cs, adaga_cs, nn_cs, n_cs, JW_cs)
	p2 = spinal_fermion_site_ops_u1_u1()
	adag_cc, a_cc, adagadag_cc, adaga_cc, nn_cc, n_cc, JW_cc = p2["+"], p2["-"], p2["++"], p2["+-"], p2["n↑n↓"], p2["n"], p2["JW"]
	@tensor adagJW_cc[1,2;3,5] := adag_cc[1,2,3,4] * JW_cc[4,5]
	ops_cc = (adag_cc, adagJW_cc, a_cc, adagadag_cc, adaga_cc, nn_cc, n_cc, JW_cc)
	for (a, b) in zip(ops_cs, ops_cc)
		@test norm(a) ≈ norm(b) atol = 1.0e-12
	end
	# test a†a == n
	for (adag, n, JW) in zip((adag_cs, adag_cc), (n_cs, n_cc), (JW_cs, JW_cc))
		ph = space(adag, 2)
		physpaces = [ph for i in 1:4]
		u = get_trivial_leg(n)
		@tensor JW′[3,1;4,2] := JW[1,2] * u[3] * conj(u[4])
		h1 = prodmpo(Float64, physpaces, [1,2,3], [JW′, JW′, adag])
		@test space_l(h1) == oneunit(ph)
		@test space_r(h1) == space_r(adag)
		h = h1 * h1'
		@test space_l(h) == oneunit(ph)
		@test space_r(h)' == oneunit(ph)
		@tensor n′[3,1;4,2] := n[1,2] * u[3] * conj(u[4])
		h′ = prodmpo(Float64, physpaces, [3], [n′])
		@test distance(h, h′) ≈ 0. atol = 1.0e-6
		# check adjoint mult
		h1 = prodmpo(Float64, physpaces, [2], [adag])
		h2 = prodmpo(Float64, physpaces, [4], [adag])'
		h = prodmpo(4, [2, 4], [adag, adag'])
		@test distance(h1 * h2, h) ≈ 0. atol = 1.0e-6
		@test distance(h2 * h1, h) ≈ 0. atol = 1.0e-6

		h1 = prodmpo(Float64, physpaces, [2,3], [adag, adag'])
		@test bond_dimension(h1) == 2
		@test space_l(h1) == oneunit(ph)
		@test space_r(h1)' == oneunit(ph)
		h2 = convert(MPO, h1')
		h = h1 + h2
		# orthogonalization
		@test !isleftcanonical(h)
		@test !isrightcanonical(h)
		h′ = leftorth!(copy(h), alg = Orthogonalize(QR(), normalize=true))
		@test isleftcanonical(h′)
		h′ = rightorth!(copy(h), alg = Orthogonalize(SVD(), normalize=true))
		@test isrightcanonical(h′)
		# deparallelisation
		h′ = leftdeparallel!(copy(h))
		@test distance(h, h′) ≈ 0. atol = 1.0e-6
		h′ = rightdeparallel!(copy(h))
		@test distance(h, h′) ≈ 0. atol = 1.0e-6
		h′ = deparallel!(copy(h))
		@test distance(h, h′) ≈ 0. atol = 1.0e-6
	end

	# test adjoint expectation
	L = 4
	ph = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
	right = Rep[U₁×SU₂]((0., 0.)=>1)
	psiB = randommps(ComplexF64, [ph for i in 1:L], D=10, right=right)
	psiA = randommps(ComplexF64, [ph for i in 1:L], D=10, right=right)

	right2 = Rep[U₁×SU₂]((0.5, 0.5)=>1)
	mpoA = randommpo(Float64, [ph for i in 1:L], D=4, right=right2)
	mpoB = randommpo(Float64, [ph for i in 1:L], D=4, right=right2)

	v1 = expectation(psiA, mpoA' * mpoB, psiB)
	v2 = dot(mpoA * psiA, mpoB * psiB)
	@test v1 ≈ v2 atol = 1.0e-6
	v3 = conj(expectation(psiB, mpoB' * mpoA, psiA))
	@test v1 ≈ v3 atol = 1.0e-6

end