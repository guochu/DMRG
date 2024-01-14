# mpo deparallelisation
function leftdeparallel!(x::MPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = 1:(length(x)-1)
		(verbosity > 3) && println("deparallelisation sweep from left to right on site: $i")
		M, Tm = leftdeparallel(x[i], (1,2,4), (3,); tol=tol, verbosity=verbosity)
		x[i] = permute(M, (1,2), (4,3))
	    @tensor tmp[-1 -2; -3 -4] := Tm[-1, 1] * x[i+1][1,-2,-3,-4]
	    x[i+1] = tmp
	end
	return x
end

function rightdeparallel!(x::MPO; tol::Real=DeparalleliseTol, verbosity::Int=0)
	for i = length(x):-1:2
	    (verbosity > 3) && println("deparallelisation sweep from right to left on site: $i")
	    Tm, M = rightdeparallel(x[i], (1,), (2,3,4); tol=tol, verbosity=verbosity)
        x[i] = permute(M, (1,2), (3,4))
        @tensor tmp[-1 -2; -3 -4] := x[i-1][-1,-2,1,-4] * Tm[1,-3]
        x[i-1] = tmp
	end
	return x
end

function deparallel!(x::MPO; kwargs...)
	r = leftdeparallel!(x; kwargs...)
	r = rightdeparallel!(x; kwargs...)
	return r
end

"""
	deparallel(x::AbstractMPO)
	reduce the bond dimension of mpo using deparallelisation
"""
deparallel(x::MPO; kwargs...) = deparallel!(copy(x); kwargs...)