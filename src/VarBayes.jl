module VarBayes

using Distributions

import Base: rand
import Distributions: logpdf

export infer, fit_mm_em, init_mm_var, fit_mm_var!, fit_cimm_var!

include("fit_mm_em.jl")
include("fit_cimm_var.jl")
include("fit_mm_var.jl")
include("tdp.jl")
include("normal_model.jl")
include("diagnormal.jl")

end