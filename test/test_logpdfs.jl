using Base.Test

using Distributions
using VarBayes

# some random positive-definite matrix
s = [
  4.43472  -1.48268   -0.95897   -1.45701;
 -1.48268   2.61408   -0.115819   0.524627;
 -0.95897  -0.115819   1.59856   -0.380318;
 -1.45701   0.524627  -0.380318   2.09062;
 ]

# random mean
m = [-0.575411,0.84235,-0.526959,0.561692]

# random test vector
x=[ -0.187018 -1.61492 -2.35163 0.0311743]'

niw = NormalInverseWishart(m, 1, s, 4)

# obtained by monte-carlo integration with 160,000,000 samples
# logpdf_model(niw, MvNormal, x, <nsamples>)
ans = -9.4153   # std. dev. : 0.0080521 / sqrt(160)

@test_approx_eq_eps logpdf(BayesNormal(niw,[]), x) ans 0.001

#----------------------------------------------
# another random positive-definite matrix
s = [
 3.44651  2.20916   1.24665   1.44052;
 2.20916  1.76731   0.983208  1.02336;
 1.24665  0.983208  1.22772   1.29937;
 1.44052  1.02336   1.29937   1.57862;
]

# random mean
m = [1.68192, 2.23138, 2.54803, 1.12245]

# random test vector
x = [0.369552 1.04611 -1.99544 0.190008]'

niw = NormalInverseWishart(m, 1, s, 6)

# TODO: try with 5x5 matrices too

# obtained by monte carlo integration with 556,000,000 samples
ans = -15.507  # std. dev: 0.36753 / sqrt(556)

@test_approx_eq_eps logpdf(BayesNormal(niw,[]), x) ans 0.1

#-------------------------------------------------
# Test gamma
require("diagnormal.jl")

# random shape parameters
alpha = 4.23
beta = 2.49

# random mean
m = 1.2

# random lambda parameter
l = 3.48

# random test number
x = Array(Float64, 1, 1)
x[1] = 2.8605

nig = NormalInverseGamma(m, l, alpha, beta)

# computed with Maxima (see normal_inverse_gamma.wxm)
ans = -2.005332101241515

@test_approx_eq_eps logpdf(BayesDiagNormal([nig]), x) ans 1e-7
