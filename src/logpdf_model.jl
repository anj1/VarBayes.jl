# given a graphical model that looks like this:
# <prior on params> ---> distribution ---> data,
# where both the prior and data are known,
# return the log-likelihood of the model.
# In other words, marginalize over the parameters.

# the general format for calling is:
# (prior, DistributionType, data)

using Distributions

# naive generic implementation of logpdf_model; using
# monte carlo integration. Applies to all distribution types
# (except for a few 'weirdos' that don't produce the right tuples)
function logpdf_model{T}(pri, ::Type{T}, x, niters)

	tsum = 0
	t = Array(Float64, niters)
	for i = 1 : niters
		prm = rand(pri)
		t[i] = sum(pdf(T(prm...), x))[1]
	end
	return log(sum_kbn(t)) - log(niters)
end
