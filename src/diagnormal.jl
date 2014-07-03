using Distributions

# A multivariate normal with a different prior for each
# coordinate.
type BayesDiagNormal{P}
  pri::Vector{P}
end

# A more easier way of initializing with mean and variance vector
# Addition params for P given in Pparams
function BayesDiagNormal{P}(::Type{P}, mu, v, Pparams)
	n = length(mu)
	n == length(v) || throw(ArgumentError("mu and v must be of equal length."))
	BayesDiagNormal([P(mu[i], v[i], Pparams...) for i in 1:n])
end

# Normal-gamma prior
function logpdf(p::BayesDiagNormal{NormalInverseGamma}, x)
	n = size(x,2)
	dim = length(p.pri)
	lp = zeros(Float64, 1, n)

	for i = 1 : dim
		pri = p.pri[i]
		μ, v0, α, β = pri.mu, pri.v0, pri.shape, pri.scale
		ratκ = α / (v0*β)
		lp .+= Distributions.logpdf(TDist(2α), sqrt(ratκ)*(x[i,:] .- μ)) .+ 0.5*log(ratκ)
	end
	lp
end


fit_mleb(p::BayesDiagNormal{NormalInverseGamma}, x, w) =
   BayesDiagNormal(
     [posterior(p.pri[i], Normal, x[i,:], w[i,:]) for i in length(p.pri)]
  )
