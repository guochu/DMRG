# arXiv:1407.1832v1 "Time-evolving a matrix product state with long-ranged interactions"
abstract type TimeEvoMPOAlgorithm <: MPSAlgorithm end

@with_kw struct WI <: TimeEvoMPOAlgorithm
	tol::Float64 = Defaults.tol
	maxiter::Int = Defaults.maxiter
end

@with_kw struct WII <: TimeEvoMPOAlgorithm
	tol::Float64 = Defaults.tol
	maxiter::Int = Defaults.maxiter
end

function get_A(x::SchurMPOTensor)
	s1, s2 = size(x)
	return x.Os[2:s1-1, 2:s2-1]
end
function get_B(x::SchurMPOTensor)
	s1, s2 = size(x)
	return x.Os[2:s1-1, end]
end
function get_C(x::SchurMPOTensor)
	s1, s2 = size(x)
	return x.Os[1, 2:s2-1]
end
get_D(x::SchurMPOTensor) = x.Os[1, end]

function _SiteW_impl(WA, WB, WC, WD)
	s1, s2 = size(WA)
	r = Matrix{Any}(undef, s1+1, s2+1)
	r[1, 1] = WD
	for l in 2:s2+1
		r[1, l] = WC[l-1]
	end
	for l in 2:s1+1
		r[l, 1] = WB[l-1]
	end
	r[2:end, 2:end] .= WA
	return SparseMPOTensor(r)
end

function timeevompo(m::SchurMPOTensor, dt::Number, alg::WI)
	WA = get_A(m)
	WB = get_B(m) .* sqrt(dt)
	WC = get_C(m) .* sqrt(dt)
	D = get_D(m)
	WD = isomorphism(storagetype(D), codomain(D), domain(D)) + dt * D
	return _SiteW_impl(WA, WB, WC, WD)
end


function timeevompo(m::SchurMPOTensor, dt::Number, alg::WII)
	s1, s2 = size(m)
	T = mpotensortype(spacetype(m), promote_type(scalartype(m), scalartype(dt)))
	WA = Matrix{T}(undef, s1-2, s2-2)
	WB = Vector{T}(undef, s1-2)
	WC = Vector{T}(undef, s2-2)
	D = convert(T, get_D(m)) 
	WD = D

	δ = dt
	for j in 2:s1-1, k in 2:s2-1
		init_1 = isometry(storagetype(D), codomain(D), domain(D))
		init = [init_1, zero(m[1, k]), zero(m[j, end]), zero(m[j, k])]

		(y, convhist) = exponentiate(1.0, RecursiveVec(init),
									 Arnoldi(; tol=alg.tol, maxiter=alg.maxiter)) do x
			out = similar(x.vecs)

			@tensor out[1][-1 -2; -3 -4] := δ * x[1][-1 1; -3 -4] * m[1, end][2 -2; 2 1] 

			@tensor out[2][-1 -2; -3 -4] := δ * x[2][-1 1; -3 -4] * m[1, end][2 -2; 2 1] 

			@tensor out[2][-1 -2; -3 -4] += sqrt(δ) * x[1][4 3; 4 -4] * m[1, k][-1 -2; -3 3] 

			@tensor out[3][-1 -2; -3 -4] := δ * x[3][-1 1; -3 -4] * m[1, end][2 -2; 2 1] 

			@tensor out[3][-1 -2; -3 -4] += sqrt(δ) * x[1][4 3; 4 -4] * m[j, end][-1 -2; -3 3]

			@tensor out[4][-1 -2; -3 -4] := δ * x[4][-1 1; -3 -4] * m[1, end][2 -2; 2 1] 

			@tensor out[4][-1 -2; -3 -4] += x[1][4 3; 4 -4] * m[j, k][-1 -2; -3 3]  

			@tensor out[4][-1 -2; -3 -4] += sqrt(δ) * x[2][4 3; -3 -4] * m[j, end][-1 -2; 4 3]  

			@tensor out[4][-1 -2; -3 -4] += sqrt(δ) * x[3][-1 4; 3 -4] * m[1, k][3 -2; -3 4] 

			return RecursiveVec(out)
		end
		convhist.converged == 0 && @warn "failed to exponentiate $(convhist.normres)"

		WA[j - 1, k - 1] = y[4]
		WB[j - 1] = y[3]
		WC[k - 1] = y[2]
		WD = y[1]

	end
	return _SiteW_impl(WA, WB, WC, WD)
end

function _W_impl(h::Vector)
	h2 = copy(h)
	h2[1] = SparseMPOTensor(h[1].Os[1:1, :], h[1].leftspaces[1:1], h[1].rightspaces, h[1].pspace)
	h2[end] = SparseMPOTensor(h[end].Os[:, 1:1], h[end].leftspaces, h[end].rightspaces[1:1], h[end].pspace)
	return MPOHamiltonian(h2)
end

timeevompo(m::MPOHamiltonian, dt::Number, alg::TimeEvoMPOAlgorithm) = _W_impl([timeevompo(mj, dt, alg) for mj in m.data])
timeevompo(m::MPOHamiltonian, dt::Number; alg::TimeEvoMPOAlgorithm = WII()) = timeevompo(m, dt, alg)