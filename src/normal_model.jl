# This file introduces several new types for multivariate normals 
# with encapsulated priors.
using Distributions

# -------------------------------------------------------------------
# A MvNormal together with a prior P
type BayesNormal{P} <: Distribution{Multivariate,Continuous}
	pri::P
	μ    # Mean; although it may not be used
end

# -------------------------------------------------------------------
# Normal-Inverse-Wishart prior
function logpdf(p::BayesNormal{NormalInverseWishart}, x)
	pri = p.pri
	κ = pri.kappa
	iwishart_logpdf(pri.mu, pri.nu, pri.Lamchol, κ/(κ+1), x) 
end

fit_mleb(p::BayesNormal{NormalInverseWishart}, x, w) = 
   BayesNormal(posterior(p.pri, MvNormal, x, w), [])

# -------------------------------------------------------------------
# Inverse-Wishart prior
function logpdf{T}(p::BayesNormal{InverseWishart}, x)
	pri = p.pri
	iwishart_logpdf(p.μ, pri.nu, pri.Psichol, 1.0, x)
end

fit_mleb(p::BayesNormal{InverseWishart}, x, w) = 
   BayesNormal{InverseWishart}(posterior(p.pri, MvNormal, x.-p.μ, w), p.μ)

# Common internals ----------------------------------------------

# log-pdf of inverse-wishart, given data (x), mean (μ), 
# wishart parameters (ν, Ψch), scale to multiply centered
# data by (scale).
function iwishart_logpdf(μ, ν, Ψch, scale, x::Matrix)
	D, n = size(x) # number of dimensions
	D = float64(D)

	ldΨ  = logdet(Ψch)
	invΨ = inv(Ψch)
	
	l = Array(Float64,n)

	y = sqrt(scale)*(x .- μ)  # center data
	for i = 1 : n
		ldΨA = logdet_with_scatter(ldΨ, invΨ, y[:,i])
		l[i] = log_iwishart_integral(ν, D, ldΨ, ldΨA) + 0.5*D*log(scale) # TODO: this could use a lot of optimization.
	end

	l
end

# compute log(|Ψ + xx'|)
# use determinant lemma: det(Ψ + uu') = (1 + u'inv(Ψ)u) det(Ψ)
logdet_with_scatter(ldΨ, invΨ, x) = log(1 + (x'*invΨ*x)[1]) + ldΨ

# n: number of data vectors
# D: number of data dimensions
# ν: confidence in covariance parameter
# ldΨ: log-det of covariance parameter
# ldΨA: log-det of covariance parameter plus data scatter matrix
log_iwishart_integral(ν, D, ldΨ, ldΨA) = 
	lgamma(0.5*(ν+1)) -
	lgamma(0.5*(ν+(1-D))) +
    0.5*( ν   *ldΨ -
       	 (ν+1)*ldΨA -
         D*1*log(pi))

# -----------------------------------------------------------