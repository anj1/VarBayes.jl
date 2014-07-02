require("fit_mm_em.jl")
require("fit_mm_var.jl")
require("tdp.jl")
require("normal_model.jl")

# priors
comppri = BayesNormal(NormalInverseWishart(zeros(4),0.1,eye(4),4),[])
mixpri  = BayesCategorical(Dirichlet(6,1.0))

# model
mm = init_mm_var(comppri, mixpri, 6)

# some data
x = cat(2, randn(4,10), randn(4,10).+3)

# fit model for a few iterations
for i = 1:3
	fit_mm_var!(mm, comppri, mixpri, x)
end

@show mm
