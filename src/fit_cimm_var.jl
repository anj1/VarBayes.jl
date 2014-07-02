# Variational estimation for
# circular shift-invariant Mixture Model (CIMM)
function logp_to_p_rot(lp)
	r = exp(lp .- maximum(lp))
	r ./ sum(sum(r,3),2)
end

function fit_cimm_var!(m::MixtureModel, comppri, mixpri, x, rskip)
	R, rem = divrem(size(x,1), rskip)
	assert(rem==0)

	ncomps = length(m.component)

	# calculate probs and also extend x by adding shifted copies of data
	ρ = Array(Float64, size(x,2), ncomps, R)
	y = Array(Float64, size(x,1), 0)
	for r = 1 : R
		shiftx = circshift(x, [rskip*(r-1), 0])
		ρ[:,:,r] = infer(m, shiftx)
		y = cat(2, y, shiftx)
	end

	s = logp_to_p_rot(ρ)

	# for each component
	cr = 1:ncomps
	for k in cr
		# Update this component
		m.component[k] = fit_mleb(comppri, y, reshape(s[:,k,:], (size(x,2)*R,)))

		# update probabilities
		for r = 1 : R
			shiftx = circshift(x, [rskip*(r-1), 0])
			ρ[:,k,r] = logpdf(m.component[k], shiftx) .+ logpdf(m.mixing, k)
		end
		s = logp_to_p_rot(ρ)
	end

	# Finally, update the mixing parameters
	# Note that when calculating the mixing parameters, all rotations
	# of the component are considered to be the same component.
	m.mixing = fit_mleb(mixpri, [cr], vec(sum(sum(s,1),3)))
end
