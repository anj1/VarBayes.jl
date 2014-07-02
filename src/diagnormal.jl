using Distributions

# A multivariate normal with a different prior for each
# coordinate.
type BayesDiagNormal{P}
  pri::Vector{P}
end

# Normal-gamma prior
function logpdf(p::BayesDiagNormal{NormalInverseGamma}, x)
	(dim, n) = size(x)
	lp = zeros(Float64, 1, n)

	for i = 1 : dim
		pri = p.pri[i]
		μ, v0, α, β,   = pri.mu, pri.v0, pri.shape, pri.scale
		ratκ = α / (v0*β)
		lp .+= Distributions.logpdf(TDist(2α), sqrt(ratκ)*(x[i,:] .- μ)) .+ 0.5*log(ratκ)
	end
	lp
end


fit_mleb(p::BayesNormal{NormalInverseGamma}, x, w) =
   BayesNormal(
     [posterior(p.pri[i], Normal, x[i,:], w[i,:]) for i in length(p.pri)]
  )
