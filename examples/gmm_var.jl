# Variational learning applied to simple gaussian mixture model.

using Distributions
using VarBayes

K = 6   # number of components

# priors
comppri = BayesNormal(NormalInverseWishart(zeros(4),0.1,eye(4),4))
mixpri  = BayesCategorical(Dirichlet(K,1.0))

# model
mm = init_mm_var(comppri, mixpri, K)

# some data
x = cat(2, randn(4,10), randn(4,10).+3)

# fit model for a few iterations
for i = 1:10
	fit_mm_var!(mm, comppri, mixpri, x)
end

# Display results (means of variational distributions).
for k = 1:K
	n = mm.mixing.pri.alpha[k]
	mu = mm.component[k].pri.mu
	@show k, n, mu
end
