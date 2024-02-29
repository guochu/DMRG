canonicalize(h::MPO; kwargs...) = canonicalize!(copy(h); kwargs...)
function canonicalize!(h::MPO; alg::Orthogonalize = Orthogonalize(SVD(), DefaultTruncation, normalize=false))
	_leftorth!(h, QR(), TK.NoTruncation(), alg.normalize)
	return rightorth!(h, alg=alg)
end

TK.leftorth!(h::MPO; alg::Orthogonalize = Orthogonalize()) = _leftorth!(h, alg.orth, alg.trunc, alg.normalize)
function _leftorth!(h::MPO, alg::QR, trunc::TruncationScheme, normalize::Bool)
	!isa(trunc, TK.NoTruncation) &&  @warn "truncation has no effect with QR"
	L = length(h)
	for i in 1:L-1
		q, r = leftorth(h[i], (1,2,4), (3,), alg = alg)
		_renormalize!(h, r, normalize)
		h[i] = permute(q, (1,2), (4,3))
		h[i+1] = @tensor tmp[1 3; 4 5] := r[1,2] * h[i+1][2,3,4,5]
	end
	_renormalize!(h, h[L], normalize)
	_renormalize_coeff!(h, normalize)
	return h
end

function _leftorth!(h::MPO, alg::SVD, trunc::TruncationScheme, normalize::Bool)
	L = length(h)
	# errs = Float64[]
	for i in 1:L-1
		u, s, v, err = stable_tsvd(h[i], (1,2,4), (3,), trunc=trunc)
		_renormalize!(h, s, normalize)
		h[i] = permute(u, (1,2), (4,3))
		r = s * v
		h[i+1] = @tensor tmp[1 3; 4 5] := r[1,2] * h[i+1][2,3,4,5]
	end
	_renormalize!(h, h[L], normalize)
	_renormalize_coeff!(h, normalize)
	return h
end


TK.rightorth!(h::MPO; alg::Orthogonalize = Orthogonalize()) = _rightorth!(h, alg.orth, alg.trunc, alg.normalize)
function _rightorth!(h::MPO, alg::QR, trunc::TruncationScheme, normalize::Bool)
	!isa(trunc, TK.NoTruncation) &&  @warn "truncation has no effect with QR"
	L = length(h)
	for i in L:-1:2
		l, q = rightorth(h[i], (1,), (2, 3, 4), alg=LQ())
		_renormalize!(h, l, normalize)
		h[i] = permute(q, (1,2), (3,4))
		h[i-1] = @tensor tmp[-1 -2; -3 -4] := h[i-1][-1, -2, 1, -4] * l[1, -3]
	end
	_renormalize!(h, h[1], normalize)
	_renormalize_coeff!(h, normalize)
	return h
end

function _rightorth!(h::MPO, alg::SVD, trunc::TruncationScheme, normalize::Bool)
	L = length(h)
	for i in L:-1:2
		u, s, v, err = stable_tsvd(h[i], (1,), (2,3,4), trunc=trunc)
		_renormalize!(h, s, normalize)
		h[i] = permute(v, (1,2), (3,4))
		u = u * s
		h[i-1] = @tensor tmp[-1 -2; -3 -4] := h[i-1][-1, -2, 1, -4] * u[1, -3]
	end
	_renormalize!(h, h[1], normalize)
	_renormalize_coeff!(h, normalize)
	return h
end
