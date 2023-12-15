# # customized truncation

# """
# 	struct MPSTruncation
# truncate singular values below ϵ first, if the remaining bond dimension is larger than dim, then truncate it below dim.
# Return the p-norm of the truncated singular values.
# """
# struct MPSTruncation <: TruncationScheme
# 	add_back::Int
# 	D::Int
# 	ϵ::Float64
# 	verbosity::Int
# end


# # MPSTruncation(D::Int, ϵ::Real; verbosity::Int=1) = MPSTruncation(D, convert(Float64, ϵ), verbosity)
# MPSTruncation(;D::Int, ϵ::Real, add_back::Int=1, verbosity::Int=1) = MPSTruncation(min(add_back, D), D, convert(Float64, ϵ), verbosity)

# compute_size(v::AbstractVector) = length(v)
# function compute_size(v::AbstractDict) 
# 	init = 0
# 	for (c, b) in v
# 		init += dim(c) * length(b)
# 	end
# 	return init
# end

# function TK._truncate!(v, trunc::MPSTruncation, p::Real = 2)
# 	n = TK._norm(v, p, 0.)

# 	verbosity = trunc.verbosity
# 	s_1 = compute_size(v)
# 	# truncate using relative cutof
# 	v, err1 = TK._truncate!(v, truncbelow(trunc.ϵ * n, trunc.add_back), p)
# 	s_2 = compute_size(v)
# 	if s_2 <= trunc.D
# 		(verbosity > 2) && @printf("sum: %4i -> %4i\n", s_1, s_2)
# 		return v, norm([err1], p)
# 	end
# 	# if still larger than D, then truncate using D
# 	v, err2 = TK._truncate!(v, truncdim(trunc.D), p)
# 	s_3 = compute_size(v)
# 	err = norm((err1, err2), p)

# 	(verbosity > 1) && @printf("sum: %4i -> %4i -> %4i, maximum %4i, truncation error: abs=%e, rel=%e\n", s_1, s_2, s_3, trunc.D, err, err/n)

# 	return v, err / n
# end

const DefaultTruncation = TruncationDimCutoff(D=Defaults.D, ϵ=Defaults.tolgauge, add_back=0)

