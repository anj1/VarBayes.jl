# Implement Dahua Lin's example of a modified GMM.
# The modified GMM's mixing weights have a Dirichlet prior,
# The covariances are all shared,
# And the mean coordinates have a univariate normal prior

# The interface provided by fit_mm_em allows us to perform
# full EM on this new model without changing anything and
# only defining some new types and fit_mle on those types.

# by Alireza Nejati

using Distributions
import Distributions:logpdf
require("fit_mm_em.jl")

# New type and function definitions ---------------------------------

# This type encapsulates a general distribution along
# with a prior over its parameters.
type BayesDist <: Distribution
	pri                    # a prior over the parameters of dist (tuple)
	dist::Distribution     # the distribution itself
end

fit_em{T<:BayesDist}(m::T, x, w) = 
  T(m.pri, fit_map(m.pri, typeof(m.dist), x, w))

logpdf{T<:BayesDist}(m::T, x) = logpdf(m.dist, x)

# Test --------------------------------------------------------------
# Now test the above code on a simple model using EM

# dimensionality of output
K = 4

# sample data
x = cat(2, randn(K,64).-3, randn(K,32).+3)

# Diagonal prior (note that we can't use IsoNormal)
v = 0.0
sig = 1.0
mu = MvNormal(fill(v,K), diagm(fill(sig,K)))

# Initial guess for components and mixture weights (random)
# note that, here, sigma is shared.
comps = Array(BayesDist, 2)
sigma = eye(4)
comps[1] = BayesDist((mu, sigma), MvNormal(randn(K), sigma))
comps[2] = BayesDist((mu, sigma), MvNormal(randn(K), sigma))

# Dirichlet prior
alpha = 2.0
mix = BayesDist(Dirichlet(fill(alpha/2,2)), Categorical([0.5, 0.5]))

m = MixtureModel(mix, comps)

# Run EM algorithm for a few iterations
for i = 1:10
	m = fit_mm_em(m, x)
end

println(m.component[1].dist.μ)
println(m.component[2].dist.μ)
println(m.component[2].dist.Σ)
println(m.mixing.dist.prob)