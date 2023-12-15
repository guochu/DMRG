println("------------------------------------")
println("|               MPS                |")
println("------------------------------------")

@testset "MPS initializer: product state and random state" begin
	VZ2 = Rep[ℤ₂](0=>1, 1=>1)
	physpaces = [VZ2 for i in 1:4]
	psi = prodmps(ComplexF64, [VZ2 for i in 1:4], [0, 1, 1, 0])
	@test length(psi) == 4
	for i in 1:length(psi)
		@test !isdual(space(psi[i], 1))
		@test !isdual(space(psi[i], 2))
		@test isdual(space(psi[i], 3))
	end
	for i in (1, length(psi)+1)
		@test !isdual(space(psi.s[i], 1))
		@test isdual(space(psi.s[i], 2))
	end
	@test psi.s[1] == one(psi.s[1])
	@test psi.s[end] == one(psi.s[end])
	@test scalartype(psi) == ComplexF64
	@test bond_dimensions(psi) == [1,1,1,1]
	@test space_l(psi) == oneunit(VZ2)
	@test space_r(psi) == oneunit(VZ2)'
	@test physical_spaces(psi) == physpaces
	@test isleftcanonical(psi)
	@test isrightcanonical(psi)
	@test !iscanonical(psi)
	@test norm(psi) ≈ 1.
	psi2 = prodmps(ComplexF64, [VZ2 for i in 1:4], [1, 0, 1, 0])
	@test norm(psi + psi2) ≈ sqrt(2)
	@test norm(psi - psi2) ≈ sqrt(2)
	@test dot(psi, psi2) ≈ 0.

	VSU2 = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
	physpaces = [VSU2 for i in 1:4]
	psi = prodmps(Float64, physpaces, [(0, 0.5), (0.5, 0), (0, 0.5), (-0.5, 0)])
	for i in 1:length(psi)
		@test !isdual(space(psi[i], 1))
		@test !isdual(space(psi[i], 2))
		@test isdual(space(psi[i], 3))
	end
	for i in (1, length(psi)+1)
		@test !isdual(space(psi.s[i], 1))
		@test isdual(space(psi.s[i], 2))
	end
	@test psi.s[1] == one(psi.s[1])
	@test psi.s[end] == one(psi.s[end])
	@test scalartype(psi) == Float64
	@test bond_dimension(psi) == 2
	@test space_l(psi) == oneunit(VSU2)
	@test space_r(psi) == oneunit(VSU2)'
	@test physical_spaces(psi) == physpaces
	@test norm(psi) ≈ 1.

	# nontrivial sector
	right = Rep[U₁×SU₂]((0, 0.5)=>1)
	psi = prodmps(Float64, [VSU2 for i in 1:3], [(0.5, 0), (-0.5, 0.), (0, 0.5)], right=right)
	@test space_r(psi) == right'
	@test norm(psi) ≈ sqrt(dim(right))
	@test !isleftcanonical(psi)
	@test !isrightcanonical(psi)
	@test !iscanonical(psi)

	psi1 = leftorth!(copy(psi), alg = Orthogonalize(QR(), normalize=false))
	@test norm(psi1) ≈ sqrt(dim(right))
	psi1 = leftorth!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	@test norm(psi1) ≈ 1.
	@test isleftcanonical(psi1)
	psi1 = rightorth!(copy(psi), alg = Orthogonalize(SVD(), normalize=false))
	@test norm(psi1) ≈ sqrt(dim(right))
	psi1 = rightorth!(copy(psi), alg = Orthogonalize(QR(), normalize=true))
	@test norm(psi1) ≈ 1.
	@test isrightcanonical(psi1)
	psi1 = canonicalize!(copy(psi), alg = Orthogonalize(QR(), normalize=false))
	@test norm(psi1) ≈ sqrt(dim(right))
	@test !iscanonical(psi1)
	psi1 = canonicalize!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	@test norm(psi1) ≈ 1.
	@test iscanonical(psi1)

	# random mps, trivial sector
	psi = randommps(ComplexF64, [VSU2 for i in 1:4], right=oneunit(right), D=4)
	@test bond_dimension(psi) <= 4
	psi1 = leftorth!(copy(psi), alg = Orthogonalize(QR(), normalize=true))
	@test isleftcanonical(psi1)
	psi1 = rightorth!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	@test isrightcanonical(psi1)
	psi1 = canonicalize!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	@test iscanonical(psi1)
	@test norm(2 * psi1) ≈ 2
	@test norm(psi1 / 2) ≈ 0.5
	@test norm(psi1 - psi1) ≈ 0. atol = 1.0e-12
	@test distance(psi, psi) ≈ 0. atol = 1.0e-12

	# random mps, nontrivial sector
	psi = randommps(ComplexF64, [VSU2 for i in 1:5], right=right, D=4)
	@test space_r(psi)' == right

	psi1 = leftorth!(copy(psi), alg = Orthogonalize(QR(), normalize=true))
	@test isleftcanonical(psi1)
	psi1 = rightorth!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	@test isrightcanonical(psi1)
	psi1 = canonicalize!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	@test iscanonical(psi1)
	@test norm(2 * psi1) ≈ 2 atol = 1.0e-12
	@test norm(psi1 / 2) ≈ 0.5 atol = 1.0e-12
	@test norm(psi1 - psi1) ≈ 0. atol = 1.0e-12
	@test dot(psi1, psi1) ≈ 1. atol = 1.0e-12
	@test distance(psi, psi) ≈ 0. atol = 1.0e-12
end

@testset "ExactMPS" begin
	VSU2 = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
	right = Rep[U₁×SU₂]((0, 0.5)=>1)
	psi = randommps(ComplexF64, [VSU2 for i in 1:4], right=oneunit(right), D=4)
	psi2 = ExactMPS(psi)
	@test norm(psi) ≈ norm(psi2) atol = 1.0e-12
	for i in 1:psi2.center - 1
		@test isleftcanonical(psi2[i]) atol = 1.0e-12
	end
	for i in psi2.center+1:length(psi2)
		@test isrightcanonical(psi2[i]) atol = 1.0e-12
	end

	psi1 = canonicalize!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	psi2 = ExactMPS(psi1)
	@test norm(psi2) ≈ 1. atol = 1.0e-12
	@test dot(psi1, psi2) ≈ 1 atol = 1.0e-12
	@test dot(psi2, psi2) ≈ 1 atol = 1.0e-12
	# nontrivial sector
	psi = randommps(ComplexF64, [VSU2 for i in 1:5], right=right, D=4)
	psi2 = ExactMPS(psi)
	@test norm(psi) ≈ norm(psi2) atol = 1.0e-12
	for i in 1:psi2.center - 1
		@test isleftcanonical(psi2[i]) atol = 1.0e-12
	end
	for i in psi2.center+1:length(psi2)
		@test isrightcanonical(psi2[i]) atol = 1.0e-12
	end

	psi1 = canonicalize!(copy(psi), alg = Orthogonalize(SVD(), normalize=true))
	psi2 = ExactMPS(psi1)
	@test norm(psi2) ≈ 1. atol = 1.0e-12
	@test dot(psi1, psi2) ≈ 1 atol = 1.0e-12
	@test dot(psi2, psi2) ≈ 1 atol = 1.0e-12
end
