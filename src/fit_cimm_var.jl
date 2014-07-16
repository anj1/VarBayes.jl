# Variational estimation for
# circular shift-invariant Mixture Model (CIMM)

# take log-probabilities, convert to probabilities and normalize.
# dims is the set of dimensions that we normalize over.
# TODO: make this more stable
function logp_to_p_rot(lp, dims)
	r = exp(lp .- maximum(lp))
	r ./ sum(r,dims)
end

function fit_cimm_var!(m::MixtureModel, comppri, mixpri, x, rskip)
	R, rem = divrem(size(x,1), rskip)
	assert(rem==0)

	ncomps = length(m.component)

	# calculate probs and also extend x by adding shifted copies of data
	# ρ[Input, Component, Rotation]
	ρ = Array(Float64, size(x,2), ncomps, R)
	y = Array(Float64, size(x,1), 0)
	for r = 1 : R
		shiftx = circshift(x, [rskip*(r-1), 0])
		ρ[:,:,r] = infer(m, shiftx) 
		y = cat(2, y, shiftx)
	end

	s = logp_to_p_rot(ρ, (2, 3))  # Normalize over component and rotation

	# for each component
	cr = 1:ncomps
	for k in cr
		# Update this component
		# todo: replace /R with something in the calculation of s
		m.component[k] = fit_mleb(comppri, y, reshape(s[:,k,:], (size(x,2)*R,))/R)

		# update probabilities
		for r = 1 : R
			shiftx = circshift(x, [rskip*(r-1), 0])
			ρ[:,k,r] = logpdf(m.component[k], shiftx) .+ logpdf(m.mixing, k)
		end
		s = logp_to_p_rot(ρ, (2, 3))
	end

	# Finally, update the mixing parameters
	# Note that when calculating the mixing parameters, all rotations
	# of the component are considered to be the same component.
	m.mixing = fit_mleb(mixpri, [cr], vec(sum(s, (1, 3))))

	s  # todo: make the other variational fitting algorithms return state as well
end
