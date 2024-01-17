

struct MPSBondView{M}
	parent::M
end

Base.getindex(psi::MPSBondView, i::Int) = getindex(psi.parent.svectors, i)
Base.firstindex(m::MPSBondView) = firstindex(m.parent.svectors)
Base.lastindex(m::MPSBondView) = lastindex(m.parent.svectors)
function Base.setindex!(m::MPSBondView, v, i::Int)
	# check_bondtensor_dir(v) || throw(SpaceMismatch())
	L = length(m.parent)
	(1 < i <= L) || throw(BoundsError(m.parent.svectors, i))
	# (space_r(m.parent[i-1])' == space_l(v) == space_l(m.parent[i])) || throw(SpaceMismatch())
	return setindex!(m.parent.svectors, v, i)
end

