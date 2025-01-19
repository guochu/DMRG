# orthogonalize mps to be left-canonical or right-canonical
abstract type MatrixProductOrthogonalAlgorithm  end

"""
	struct MatrixProductOrthogonalize{A<:Union{QR, SVD}, T<:TruncationScheme}
"""
struct Orthogonalize{A<:Union{QR, SVD}, T<:TruncationScheme} <: MatrixProductOrthogonalAlgorithm
	orth::A
	trunc::T
	normalize::Bool
end
Orthogonalize(a::Union{QR, SVD}, trunc::TruncationScheme; normalize::Bool=false) = Orthogonalize(a, trunc, normalize)
Orthogonalize(a::Union{QR, SVD}; trunc::TruncationScheme=TK.NoTruncation(), normalize::Bool=false) = Orthogonalize(a, trunc, normalize)
Orthogonalize(; alg::Union{QR, SVD} = SVD(), trunc::TruncationScheme=TK.NoTruncation(), normalize::Bool=false) = Orthogonalize(alg, trunc, normalize)


# function rightcanonicalize!(psi::AbstractMPS; normalize::Bool=false, trunc::TruncationScheme = NoTruncation())
# 	leftorth!(psi, alg = QRFact())
# 	errs = rightorth!(psi, alg=SVDFact(trunc=trunc, normalize=normalize))
# 	return errs
# end
function rightcanonicalize!(psi::MPS; alg::Orthogonalize = Orthogonalize(SVD(), DefaultTruncation, normalize=false))
	_leftorth!(psi, QR(), TK.NoTruncation(), alg.normalize)
	 return rightorth!(psi, alg=alg)
end

"""
	canonicalize!(psi::MPS; kwargs...)
Prepare the MPS into right-canonical form with correct Schmidt numbers.
Internally ths function will do a left to right sweep and then a right to
left sweep.

Kyward arguments:
* normalize=false: normalize the MPS or not
* trunc=NoTruncation(): truncation strategy
"""
canonicalize!(psi::MPS; kwargs...) = rightcanonicalize!(psi; kwargs...)
canonicalize(psi::MPS; kwargs...) = canonicalize!(deepcopy(psi); kwargs...)

TK.leftorth!(psi::MPS; alg::Orthogonalize = Orthogonalize()) = _leftorth!(psi, alg.orth, alg.trunc, alg.normalize)
function _leftorth!(psi::MPS, alg::QR, trunc::TruncationScheme, normalize::Bool)
	!isa(trunc, TK.NoTruncation) &&  @warn "truncation has no effect with QR"
	L = length(psi)
	for i in 1:L-1
		# println("i = ", isempty(blocks(psi[i])), " ", isempty(blocksectors(domain(psi[i]))))
		# psi_i = psi[i]
		# if isempty(blocks(psi_i))
		# 	dims = TK.SectorDict{sectortype(psi_i), Int}()
		# 	W = spacetype(psi_i)(dims)
		# 	q = similar(psi_i, codomain(psi_i) ← W)
		# 	r = similar(psi_i, W ← domain(psi_i))
		# else
		# 	q, r = leftorth!(psi[i], alg = alg)
		# end
		q, r = leftorth!(psi[i], alg = alg)
		_renormalize!(psi, r, normalize)
		psi[i] = q
		psi[i + 1] = @tensor tmp[1 3; 4] := r[1,2] * psi[i+1][2,3,4]
	end
	_renormalize!(psi, psi[L], normalize)
	_renormalize_coeff!(psi, normalize)
	return psi
end

# will always use stable SVD
function _leftorth!(psi::MPS, alg::SVD, trunc::TruncationScheme, normalize::Bool)
	L = length(psi)
	# errs = Float64[]
	for i in 1:L-1
		u, s, v, err = stable_tsvd!(psi[i], trunc=trunc)
		_renormalize!(psi, s, normalize)
		psi[i] = u
		v2 = s * v
		psi[i+1] = @tensor tmp[-1 -2; -3] := v2[-1, 1] * psi[i+1][1,-2,-3]
		psi.s[i+1] = s
		# push!(errs, err)
	end
	_renormalize!(psi, psi[L], normalize)
	_renormalize_coeff!(psi, normalize)
	return psi
end


TK.rightorth!(psi::MPS; alg::Orthogonalize = Orthogonalize()) = _rightorth!(psi, alg.orth, alg.trunc, alg.normalize)
function _rightorth!(psi::MPS, alg::QR, trunc::TruncationScheme, normalize::Bool)
	!isa(trunc, TK.NoTruncation) &&  @warn "truncation has no effect with QR"
	L = length(psi)
	for i in L:-1:2
		l, q = rightorth(psi[i], (1,), (2, 3), alg=LQ())
		_renormalize!(psi, l, normalize)
		psi[i] = permute(q, (1,2), (3,))
		psi[i-1] = @tensor tmp[1 2; 4] := psi[i-1][1,2,3] * l[3,4] 
	end
	_renormalize!(psi, psi[1], normalize)
	_renormalize_coeff!(psi, normalize)
	return psi
end

function _rightorth!(psi::MPS, alg::SVD, trunc::TruncationScheme, normalize::Bool)
	L = length(psi)
	# errs = Float64[]
	for i in L:-1:2
		u, s, v, err = stable_tsvd(psi[i], (1,), (2, 3), trunc=trunc)
		_renormalize!(psi, s, normalize)
		psi[i] = permute(v, (1,2), (3,))
		u2 = u * s
		psi[i-1] = @tensor tmp[-1 -2; -3] := psi[i-1][-1, -2, 1] * u2[1, -3]
		psi.s[i] = s
		# push!(errs, err)
	end
	_renormalize!(psi, psi[1], normalize)
	_renormalize_coeff!(psi, normalize)
	return psi
end

function _renormalize!(psi, r, normalize)
	normalize && LinearAlgebra.normalize!(r)
end
# function _renormalize!(psi, r, normalize)
# 	nr = norm(r)
# 	if nr != zero(nr)
# 		if !normalize
# 			rescaling!(psi, nr)
# 		end
# 		r = rmul!(r, 1/nr)	
# 	end
# end
function _renormalize_coeff!(psi, normalize)
	# normalize && (psi.coeff[] = 1)
end
