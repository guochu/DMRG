

function boson_site_ops_u1(d::Int)
	@assert d > 1
	ph = Rep[U₁](i-1=>1 for i in 1:d)
	vacuum = oneunit(ph)
	adag = TensorMap(zeros, vacuum ⊗ ph ← Rep[U₁](1=>1) ⊗ ph)
	for i in 1:d-1
		blocks(adag)[Irrep[U₁](i)] = sqrt(i) * ones(1,1)
	end
	a = TensorMap(zeros, vacuum ⊗ ph ← Rep[U₁](-1=>1) ⊗ ph )
	for i in 1:d-1
		blocks(a)[Irrep[U₁](i-1)] = sqrt(i) * ones(1, 1)
	end
	n = TensorMap(zeros, ph ← ph)
	for i in 1:d-1
		blocks(n)[Irrep[U₁](i)] = i * ones(1, 1)
	end
	return Dict("+"=>adag, "-"=>a, "n"=>n)
end

function spin_site_ops_u1()
    ph = Rep[U₁](0=>1, 1=>1)
    vacuum = oneunit(ph)
    σ₊ = TensorMap(zeros, vacuum ⊗ ph ← Rep[U₁](1=>1) ⊗ ph)
    blocks(σ₊)[Irrep[U₁](1)] = ones(1, 1)
    σ₋ = TensorMap(zeros, vacuum ⊗ ph ← Rep[U₁](-1=>1) ⊗ ph)
    blocks(σ₋)[Irrep[U₁](0)] = ones(1, 1)
    σz = TensorMap(ones, ph ← ph)
    blocks(σz)[Irrep[U₁](0)] = -ones(1, 1)
    return Dict("+"=>σ₊, "-"=>σ₋, "z"=>σz)
end

"""
	The convention is that the creation operator on the left of the annihilation operator

By convention space_l of all the operators are vacuum
"""
function spinal_fermion_site_ops_u1_su2()
	ph = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
	bh = Rep[U₁×SU₂]((0.5, 0.5)=>1)
	vh = oneunit(ph)
	adag = TensorMap(zeros, Float64, vh ⊗ ph ← bh ⊗ ph)
	blocks(adag)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1,1)
	blocks(adag)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = sqrt(2) * ones(1,1) 

	bh = Rep[U₁×SU₂]((-0.5, 0.5)=>1)
	a = TensorMap(zeros, Float64, vh ⊗ ph ← bh ⊗ ph)
	blocks(a)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1,1)
	blocks(a)[Irrep[U₁](-0.5) ⊠ Irrep[SU₂](0)] = -sqrt(2) * ones(1,1) 


	onsite_interact = TensorMap(zeros, Float64, ph ← ph)
	blocks(onsite_interact)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = ones(1, 1)

	JW = TensorMap(ones, Float64, ph ← ph)
	blocks(JW)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = -ones(1, 1)

	# adagJW = TensorMap(zeros, Float64, vh ⊗ ph ← bh ⊗ ph)
	# blocks(adagJW)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1,1)
	# blocks(adagJW)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = -sqrt(2) * ones(1,1) 

	# hund operators
	# c↑† ⊗ c↓†
	bhr = Rep[U₁×SU₂]((1, 0)=>1)
	adagadag = TensorMap(ones, Float64, vh ⊗ ph ← bhr ⊗ ph)

	# c↑† ⊗ c↓, this is a spin 1 sector operator!!!
	bhr = Rep[U₁×SU₂]((0, 1)=>1)
	adaga = TensorMap(zeros, Float64, vh ⊗ ph ← bhr ⊗ ph)
	blocks(adaga)[Irrep[U₁](0) ⊠ Irrep[SU₂](0.5)] = ones(1, 1) * (-sqrt(3) / 2)

	n = TensorMap(ones, Float64, ph ← ph)
	blocks(n)[Irrep[U₁](-0.5) ⊠ Irrep[SU₂](0)] = zeros(1, 1)
	blocks(n)[Irrep[U₁](0.5) ⊠ Irrep[SU₂](0)] = 2 * ones(1, 1)

	return Dict("+"=>adag, "-"=>a, "++"=>adagadag, "+-"=>adaga, "n↑n↓"=>onsite_interact, "JW"=>JW, "n"=>n)
end

function spinal_fermion_site_ops_u1_u1()
	ph = Rep[U₁×U₁]((1, 1)=>1, (1,0)=>1, (0,1)=>1, (0,0)=>1)
	vacuum = oneunit(ph)

	# adag
	adagup = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((1,0)=>1) ⊗ ph )
	blocks(adagup)[Irrep[U₁](1) ⊠ Irrep[U₁](0)] = ones(1,1)
	blocks(adagup)[Irrep[U₁](1) ⊠ Irrep[U₁](1)] = ones(1,1)

	adagdown = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((0,1)=>1) ⊗ ph)
	blocks(adagdown)[Irrep[U₁](0) ⊠ Irrep[U₁](1)] = ones(1,1)
	blocks(adagdown)[Irrep[U₁](1) ⊠ Irrep[U₁](1)] = -ones(1,1)

	adag = cat(adagup, adagdown, dims=3)

	# a
	aup = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((-1,0)=>1) ⊗ ph)
	blocks(aup)[Irrep[U₁](0) ⊠ Irrep[U₁](0)] = ones(1,1)
	blocks(aup)[Irrep[U₁](0) ⊠ Irrep[U₁](1)] = ones(1,1)

	adown = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((0,-1)=>1) ⊗ ph)
	blocks(adown)[Irrep[U₁](0) ⊠ Irrep[U₁](0)] = ones(1,1)
	blocks(adown)[Irrep[U₁](1) ⊠ Irrep[U₁](0)] = -ones(1,1)

	a = cat(aup, - adown, dims=3)

	# hund operators
	adagadag = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((1,1)=>1) ⊗ ph)
	blocks(adagadag)[Irrep[U₁](1) ⊠ Irrep[U₁](1)] = ones(1, 1)

	# c↑† ⊗ c↓, this is a spin 1 sector operator!!!
	up = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((1,-1)=>1) ⊗ ph)
	blocks(up)[Irrep[U₁](1) ⊠ Irrep[U₁](0)] = ones(1,1) / (-sqrt(2))
	middle = TensorMap(zeros, Float64, vacuum ⊗ ph ← vacuum ⊗ ph )
	blocks(middle)[Irrep[U₁](1) ⊠ Irrep[U₁](0)] = 0.5 * ones(1,1)
	blocks(middle)[Irrep[U₁](0) ⊠ Irrep[U₁](1)] = -0.5 * ones(1,1)
	down = TensorMap(zeros, Float64, vacuum ⊗ ph ← Rep[U₁×U₁]((-1,1)=>1) ⊗ ph)
	blocks(down)[Irrep[U₁](0) ⊠ Irrep[U₁](1)] = ones(1,1) / sqrt(2)
	adaga = cat(cat(up, middle, dims=3), down, dims=3)

	onsite_interact = TensorMap(zeros, Float64, ph ← ph)
	blocks(onsite_interact)[Irrep[U₁](1) ⊠ Irrep[U₁](1)]= ones(1,1)

	JW = TensorMap(ones, Float64, ph ← ph)
	blocks(JW)[Irrep[U₁](1) ⊠ Irrep[U₁](0)] = -ones(1, 1)
	blocks(JW)[Irrep[U₁](0) ⊠ Irrep[U₁](1)] = -ones(1, 1)

	occupy = TensorMap(ones, Float64, ph ← ph)
	blocks(occupy)[Irrep[U₁](0) ⊠ Irrep[U₁](0)] = zeros(1, 1)
	blocks(occupy)[Irrep[U₁](1) ⊠ Irrep[U₁](1)] = 2 * ones(1, 1)
	return Dict("+"=>adag, "-"=>a, "++"=>adagadag, "+-"=>adaga, "n↑n↓"=>onsite_interact, 
		"JW"=>JW, "n"=>occupy)
end


# models

function nn_mpoham(hz, J, Jzz, p)
	sp, sm, z = p["+"], p["-"], p["z"]
	return MPOHamiltonian([fromABCD(C=[2*Ji*sp, 2*Ji*sm, Jzzi*z], B= [sp', sm', z], D=hzi*z) for (hzi, Ji, Jzzi) in zip(hz, J, Jzz)])
end
function nn_ham(hz, J, Jzz, p)
	L = length(hz)
	sp, sm, z = p["+"], p["-"], p["z"]
	mpo = prodmpo(L, [1], [hz[1] * z])
	for i in 2:L
		mpo += prodmpo(L, [i], [hz[i] * z])
	end
	compress!(mpo)
	for i in 1:L-1
		mpo += prodmpo(L, [i, i+1], [2*J[i] * sp, sp'])
		mpo += prodmpo(L, [i, i+1], [2*J[i] * sm, sm'])
		mpo += prodmpo(L, [i, i+1], [Jzz[i]*z, z])
		compress!(mpo)
	end
	return mpo
end






function initial_state_u1_su2(::Type{T}, L) where {T<:Number}
	physpace = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)

	init_state = [(-0.5, 0) for i in 1:L]
	for i in 2:2:L
		init_state[i] = (0.5, 0)
	end
	n = sum([item[1] for item in init_state])
	n2 = 0
	right = Rep[U₁×SU₂]((n, 0)=>1)
	state = prodmps(T, physpace, init_state, right=right )

	return state
end


function initial_state_u1_u1(::Type{T}, L) where {T<:Number}
	physpace = Rep[U₁×U₁]((0, 0)=>1, (0, 1)=>1, (1, 0)=>1, (1, 1)=>1)

	init_state = [(0, 0) for i in 1:L]
	for i in 2:2:L
		init_state[i] = (1, 1)
	end
	n1 = sum([item[1] for item in init_state])
	n2 = sum([item[2] for item in init_state])

	right = Rep[U₁×U₁]((n1, n2)=>1)
	state = prodmps(T, physpace, init_state, right=right )
	return state
end


function hubbard_ladder(L, J1, J2, U, p)
	adag, pp, nn, JW, n = p["+"], p["++"], p["n↑n↓"], p["JW"], p["n"]

	@tensor adagJW[1,2;3,5] := adag[1,2,3,4] * JW[4,5]
	a = adag'

	mpo = prodmpo(L, [1], [U * nn])
	for i in 2:L
		mpo += prodmpo(L, [i], [U * nn])
	end
	compress!(mpo)
	for i in 1:L-1
		tmp = prodmpo(L, [i, i+1], [-J1 * adagJW, a])
		mpo += tmp
		mpo += tmp'
		compress!(mpo)
	end

	for i in 1:L-2
		tmp = prodmpo(L, [i, i+1, i+2], [-J2*adagJW, JW, a])
		mpo += tmp
		mpo += tmp'
		compress!(mpo)
	end

	observers = [prodmpo(L, [i], [n]) for i in 1:L]

	return mpo, observers
end

max_error(a::Vector, b::Vector) = maximum(abs.(a - b))

get_trivial_leg(m::AbstractTensorMap) = TensorMap(ones,scalartype(m),oneunit(space(m,1)))

# function _add_legs(m::AbstractTensorMap{S, 1, 1}) where {S <: ElementarySpace}
# 	util=get_trivial_leg(m)
# 	@tensor m4[-1 -2; -3 -4] := util[-1] * m[-2, -4] * conj(util[-3])
# 	return m4
# end

function do_dmrg(dmrg, alg)
	dmrg_sweeps = 10
	# Evals, delta = compute!(dmrg, alg)
	Evals = Float64[]
	for i in 1:dmrg_sweeps
		Evals, delta = sweep!(dmrg, alg)
	end
	return Evals[end]
end
