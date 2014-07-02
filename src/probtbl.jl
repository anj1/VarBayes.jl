# A ProbTbl is a table of joint probabilities of N discrete random variables.
# The nice thing about it is that it can be indexed with symbols directly.
# So, for instance, if p is a ProbTbl of the joint distribution p(X,Y),
# Then p[:X] gives the probability marginalized over Y. A nice property is
# that no matter how you slice ProbTbls, the resulting arrays always sum to 1.

type ProbTbl{T,N}
	varsyms::Dict{Symbol,Int}
	tbl::Array{T,N}
end

a = ProbTbl([:X => 1, :Y => 2], rand(4,5))

function getindex{T,N}(p::ProbTbl{T,N},s::Symbol)
  dim = p.varsyms[s]
  a = p.tbl
  for i = 1 : N
    if i==dim; continue; end
    a = i < dim ? sum(a,1) : sum(a,2)
  end
  a /= sum(a)
end

# TODO: Tuples of symbols
function getindex{T,N}(p::ProbTbl{T,N},s1::Symbol,s2::Symbol)
  throw("Unimplemented functionality.")
end

a[:X]
a[:X,:Y]