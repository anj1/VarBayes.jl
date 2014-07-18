# Variational method for fitting bayesian mixture model
# Note that the definition of a normal mixture model and a Bayesian
# mixture model are identical; the difference is that the component
# distributions for a Bayesian mixture model have a prior explicitly
# included.

# to get variational dirichlet processes working, we make
# two modifications to the EM procedure in fit_mm_em.jl :
# 1. Updates can't happen all at once; each variable is updated separately.
# 2. Updates are not MLE updates. This is because a prior
#    is imposed since we are fitting a Bayesian model. The prior that is
#    used is the one before the data have been seen. These are comppri
#    and mixpri.
function fit_mm_var!(m::MixtureModel, comppri, mixpri, x)
	# calculate probs
	ρ = infer(m, x)
	r = logp_to_p(ρ, 2)

	# for each component
	ncomps = length(m.component)
	cr = 1:ncomps
	for k in cr
		# Update this component
		m.component[k] = fit_mleb(comppri, x, r[:,k])

		# update probabilities
		ρ[:,k] = logpdf(m.component[k], x) .+ logpdf(m.mixing, k)
		r = logp_to_p(ρ, 2)
	end

	# Finally, update the mixing parameters
	m.mixing = fit_mleb(mixpri, [cr], vec(sum(r,1)))

	ρ
end

# given component and mixing priors,
# generates an initial mixture model.
# K: number of components.
# Note: K must be equal to the number of components that mixpri has.
init_mm_var(comppri, mixpri, K) = MixtureModel(mixpri, [comppri for i = 1:K])
