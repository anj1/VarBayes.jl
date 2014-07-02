# A truncated Dirichlet process is a Infinite Dirichlet process
# from which samples will always be a vector with zeros after
# some K.

# A truncated Dirichlet process is actually identical to a 
# Dirichlet distribution. To see this, look at sort(rand(d)),
# where d is a Dirichlet distribution --- the result is a 
# stick-breaking process. Due to exchangeability, we don't
# have to sort the numbers when simply using as a prior.

# ----------------------------------------------------------------
# A Dirichlet-multinomial distribution
type BayesCategorical <: Distribution{Univariate,Discrete}
	pri::Dirichlet
end

# Functions defined on this distribution (rand, logpdf, fit_mleb)
Distributions.Random.rand(p::BayesCategorical) =
    rand(Categorical(rand(p.pri)))

function logpdf(p::BayesCategorical, k::Int)
	A = p.pri.alpha0
	α_k = p.pri.alpha[k]
	# TODO: this can be simplified to be much faster
	return lgamma(A) - lgamma(A+1) + lgamma(α_k+1) - lgamma(α_k)
end

# TODO: a much faster version of this is possible
logpdf(p::BayesCategorical, z::Vector{Int}) = [logpdf(p, k) for k in z]

fit_mleb(p::BayesCategorical, x, w) =
    BayesCategorical(posterior(p.pri, Categorical, x, w))