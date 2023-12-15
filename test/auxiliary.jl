println("------------------------------------")
println("auxiliary")
println("------------------------------------")


@testset "Deparallel" begin
	a = rand(20, 10) .* 1.0e-14
	b, c = leftdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test isempty(b)
	@test isempty(c)

	a = rand(20, 10) .* 1.0e-14
	b, c = rightdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test isempty(b)
	@test isempty(c)


	a = randn(8, 5)
	a[:, 3] .= a[:, 1]
	a[:, 5] .= a[:, 1]
	a[:, 4] .= a[:, 2] .+ 3.4
	b, c = leftdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test size(b, 2) == size(c, 1)
	@test size(b, 2) <= 3
	@test b * c ≈ a

	a = randn(8, 5)
	b, c = rightdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test size(b, 2) == size(c, 1)
	@test b * c ≈ a
	
	a = randn(5, 10)
	a[1, :] .= 1
	a[4, :] .= 1
	a[3, :] .= 1 + 1.0e-14
	a[5, :] .= 1	
	b, c = rightdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test size(b, 2) == size(c, 1)
	@test size(b, 2) <= 2
	@test b * c ≈ a

	VU₁ = (U1Space(0=>1, 1=>2, -1=>2),
        U1Space(0=>3, 1=>1, -1=>1),
        U1Space(0=>2, 1=>2, -1=>1),
        U1Space(0=>1, 1=>2, -1=>3))
	V1, V2, V3, V4 = VU₁
	a = TensorMap(randn, ComplexF64, V1⊗V2←V3⊗V4)
	b, c = leftdeparallel(a)
	@test b * c ≈ a
	b, c = rightdeparallel(a)
	@test b * c ≈ a

	a = TensorMap(randn, ComplexF64, V1⊗V2←V3⊗V4) * 1.0e-14
	b, c = leftdeparallel(a)
	@test dim(b) == dim(c) == 0
	b, c = rightdeparallel(a)
	@test dim(b) == dim(c) == 0

end
