using VarBayes

# priors
comppri = BayesNormal(NormalInverseWishart(randn(4),0.1,eye(4),4),[])
mixpri  = BayesCategorical(Dirichlet(6,1.0))

# model
mm = init_mm_var(comppri, mixpri, 6)

# some data
r1 = 0.1*rand(4,10)
r2 = 0.1*rand(4,10)
x = cat(2, r2.+[5,0,5,0], r1.+[5,5,0,0], r1.+[0,5,5,0])

# fit model for a few iterations
for i = 1:30
  fit_cimm_var!(mm, comppri, mixpri, x, 1)
end

@show mm
