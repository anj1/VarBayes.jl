# test EM on gaussian mixture
# by Alireza Nejati

require("fit_mm_em.jl")

# Data: two clusters separated by a distance of 6
x = cat(2, randn(4,64).-3, randn(4,32).+3)

# Initial guess for components and mixture weights (random)
sigma = 0.1*cov(x')   # empirical guesstimate
comps = [MvNormal(randn(4),sigma), MvNormal(randn(4),sigma)]
m     = MixtureModel(Categorical([0.5, 0.5]), comps)

# Run EM algorithm for a few iterations
for i = 1:10
	m = fit_mm_em(m, x)
end

show(m.component)
println(m.mixing)