# Variational learning applied to circular-invariant gaussian mixture model.

using Distributions
using VarBayes

K = 6   # number of components

# priors
comppri = BayesNormal(NormalInverseWishart(randn(4),0.1,eye(4),4))
mixpri  = BayesCategorical(Dirichlet(K,10.0))

# model
mm = init_mm_var(comppri, mixpri, K)

# some data
r1 = rand(4,10)
r2 = rand(4,10)
x = cat(2, r2.+[5,0,5,0], r1.+[5,5,0,0], r1.+[0,5,5,0])

# fit model for several iterations
for i = 1:10
  fit_cimm_var!(mm, comppri, mixpri, x, 1)
end

# Display results (means of variational distributions).
for k = 1:K
	n = mm.mixing.pri.alpha[k]
	mu = mm.component[k].pri.mu
	@show k, n, mu
end
