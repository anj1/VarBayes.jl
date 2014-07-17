# Generic EM algorithm for mixtures
# by Alireza Nejati

using Distributions

# A 'closed' mixture model defining a full generative model
type MixtureModel{T,M} <: Distribution
	mixing::M              # Distribution over mixing components
	component::Vector{T}   # Individual component distributions
end

# This returns, for a model and observations x,
# the distribution over latent variables.
function infer(m::MixtureModel, x)
	K = length(m.component)  # number of mixture components
	N = size(x,2)            # number of data points

	lq = Array(Float64, N, K)
	for k = 1:K
		lq[:,k] = logpdf(m.component[k], x) .+ logpdf(m.mixing, k)
	end
	return lq
end

# take log-probabilities, convert to probabilities and normalize.
# dims is the set of dimensions that we normalize over.
# TODO: make this more stable
function logp_to_p(lp, dims)
	r = exp(lp .- maximum(lp))
	r ./ sum(r,dims)
end

function fit_mm_em{T,M}(m::MixtureModel{T,M}, x)
	# Expectation step
	lq = infer(m, x)

	# Normalize log-probability and convert to probability
	q = logp_to_p(lq, 2)

	# Maximization step
	cr = 1:length(m.component)
	comps = [fit_em(m.component[k], x, q[:,k]) for k = cr]
	mix   =  fit_em(m.mixing, [cr], vec(sum(q,1)))

	MixtureModel(mix, comps)
end

# 'fallback' function
fit_em(m::Distribution, x, w) = fit_mle(typeof(m), x, w)
