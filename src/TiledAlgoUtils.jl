# Common functions for Tiled reconstruction algorithms

"""
Determine an extent for the rapidity tiles, by binning in 
rapidity and collapsing the outer bins until they have about
1/4 the number of particles as the maximum bin. This is the 
heuristic which is used by FastJet.
"""
function determine_rapidity_extent(_eta::Vector{T}) where T <: AbstractFloat
	length(_eta) == 0 && return 0.0, 0.0

	nrap = 20
	nbins = 2 * nrap
	counts = zeros(Int, nbins)

	# Get the minimum and maximum rapidities and at the same time bin
	# the multiplicities as a function of rapidity to help decide how
	# far out it's worth going
	minrap = floatmax(T)
	maxrap = -floatmax(T)
	ibin = 0
	for y in _eta
		minrap = min(minrap, y)
		maxrap = max(maxrap, y)

		# Bin in rapidity to decide how far to go with the tiling.
		# The bins go from ibin=1 (rap=-infinity..-19)
		# to ibin = nbins (rap=19..infinity for nrap=20)
		ibin = clamp(1 + unsafe_trunc(Int, y + nrap), 1, nbins)
		@inbounds counts[ibin] += 1
	end

	# Get the busiest bin
	max_in_bin = maximum(counts)

	# Now find minrap, maxrap such that edge bin never contains more
	# than some fraction of busiest, and at least a few particles; first do
	# it from left. NB: the thresholds chosen here are largely
	# guesstimates as to what might work.
	#
	# 2014-07-17: in some tests at high multiplicity (100k) and particles going up to
	#             about 7.3, anti-kt R=0.4, we found that 0.25 gave 20% better run times
	#             than the original value of 0.5.
	allowed_max_fraction = 0.25

	# The edge bins should also contain at least min_multiplicity particles
	min_multiplicity = 4

	# now calculate how much we can accumulate into an edge bin
	allowed_max_cumul = floor(max(max_in_bin * allowed_max_fraction,
		min_multiplicity))

	# make sure we don't require more particles in a bin than max_in_bin
	allowed_max_cumul = min(max_in_bin, allowed_max_cumul)

	# start scan over rapidity bins from the "left", to find out minimum rapidity of tiling
	cumul_lo = 0.0
	ibin_lo = 1
	while ibin_lo <= nbins
		@inbounds cumul_lo += counts[ibin_lo]
		if cumul_lo >= allowed_max_cumul
			minrap = max(minrap, ibin_lo - nrap - 1)
			break
		end
		ibin_lo += 1
	end
	@assert ibin_lo != nbins # internal consistency check that you found a bin

	# then do it from "right", to find out maximum rapidity of tiling
	cumul_hi = 0.0
	ibin_hi = nbins
	while ibin_hi >= 1
		@inbounds cumul_hi += counts[ibin_hi]
		if cumul_hi >= allowed_max_cumul
			maxrap = min(maxrap, ibin_hi - nrap)
			break
		end
		ibin_hi -= 1
	end
	@assert ibin_hi >= 1 # internal consistency check that you found a bin

	# consistency check
	@assert ibin_hi >= ibin_lo

	minrap, maxrap
end

"""
Setup the tiling parameters for this recontstruction
"""
function setup_tiling(_eta::Vector{T}, Rparam::AbstractFloat) where T <: AbstractFloat
	# First decide tile sizes (with a lower bound to avoid huge memory use with
	# very small R)
	tile_size_eta = max(0.1, Rparam)

	# It makes no sense to go below 3 tiles in phi -- 3 tiles is
	# sufficient to make sure all pair-wise combinations up to pi in
	# phi are possible
	n_tiles_phi   = max(3, floor(Int, 2π / tile_size_eta))
	tile_size_phi = 2π / n_tiles_phi # >= Rparam and fits in 2pi

	tiles_eta_min, tiles_eta_max = determine_rapidity_extent(_eta)

	# now adjust the values
	tiles_ieta_min = floor(Int, tiles_eta_min / tile_size_eta)
	tiles_ieta_max = floor(Int, tiles_eta_max / tile_size_eta) #FIXME shouldn't it be ceil ?
	tiles_eta_min = tiles_ieta_min * tile_size_eta
	tiles_eta_max = tiles_ieta_max * tile_size_eta
	n_tiles_eta = tiles_ieta_max - tiles_ieta_min + 1

	tiling_setup = TilingDef(tiles_eta_min, tiles_eta_max,
		tile_size_eta, tile_size_phi,
		n_tiles_eta, n_tiles_phi,
		tiles_ieta_min, tiles_ieta_max)

	tiling_setup
end

"""
Return the geometric distance between a pair of (eta,phi) coordinates
"""
geometric_distance(eta1::AbstractFloat, phi1::AbstractFloat, eta2::AbstractFloat, phi2::AbstractFloat) = begin
	δeta = eta2 - eta1
	δphi = π - abs(π - abs(phi1 - phi2))
	return δeta * δeta + δphi * δphi
end

"""
Return the dij metric distance between a pair of pseudojets
"""
get_dij_dist(nn_dist, kt2_1, kt2_2, R2) = begin
	if kt2_2 == 0.0
		return kt2_1 * R2
	end
	return nn_dist * min(kt2_1, kt2_2)
end

"""
Return the tile coordinates of an (eta, phi) pair
"""
get_tile(tiling_setup::TilingDef, eta::AbstractFloat, phi::AbstractFloat) = begin
	# The eta clamp is necessary as the extreme bins catch overflows for high abs(eta)
	ieta = clamp(floor(Int, (eta - tiling_setup._tiles_eta_min) / tiling_setup._tile_size_eta), 1, tiling_setup._n_tiles_eta)
	# The phi clamp should not really be necessary, as long as phi values are [0,2π)
	iphi = clamp(floor(Int, 1 + (phi / 2π) * tiling_setup._n_tiles_phi), 1, tiling_setup._n_tiles_phi)
	ieta, iphi
end

"""
Map an (η, ϕ) pair into a linear index, which is much faster "by hand" than using
the LinearIndices construct (like x100, which is bonkers, but there you go...)
"""
get_tile_linear_index(tiling_setup::TilingDef, i_η::Int, i_ϕ::Int) = begin
	return tiling_setup._n_tiles_eta * (i_ϕ-1) + i_η
end

"""
Map a linear index to a tuple of (η, ϕ) - again this is very much faster than
using CartesianIndices
"""
get_tile_cartesian_indices(tiling_setup::TilingDef, index::Int) = begin
	return (rem(index-1, tiling_setup._n_tiles_eta)+1, div(index-1, tiling_setup._n_tiles_eta)+1)
end

"""
Iterator for the indexes of rightmost tiles for a given Cartesian tile index
	- These are the tiles above and to the right of the given tile (X=yes, O=no)
		XXX
		O.X
		OOO
	- η coordinate must be in range, ϕ coordinate wraps

"""
struct rightmost_tiles
    n_η::Int		# Number of η tiles
    n_ϕ::Int		# Number of ϕ tiles
    start_η::Int	# Centre η tile coordinate
    start_ϕ::Int	# Centre ϕ tile coordinate
end

function Base.iterate(t::rightmost_tiles, state=1)
    mapping = ((-1,-1), (-1,0), (-1,1), (0,1))
    if t.start_η == 1 && state == 1
        state = 4
    end
    while state <= 4
        η = t.start_η + mapping[state][1]
        ϕ = t.start_ϕ + mapping[state][2]
        if ϕ > t.n_ϕ
            ϕ = 1
        elseif ϕ < 1
            ϕ = t.n_ϕ
        end
        return (η,ϕ), state+1
    end
    return nothing
end

"""
Populate tiling structure with our initial jets and setup neighbour tile caches
"""
function populate_tiles!(tile_jets::Array{TiledJetSoA, 2}, tiling_setup::TilingDef,
	flat_jets::FlatJetSoA, R2::AbstractFloat)
	# This is a special case, where the initial particles are all
	# "linear" in the flat_jets structure, so we scan through that
	# and match each jet to a tile, so that we can assign correct size
	# vectors in the tiled jets structure
	tile_jet_count = Array{Vector{Int}, 2}(undef, tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi)
	# Using fill() doesn't work as we fill all tiles with the same vector!
	@inbounds for itile in eachindex(tile_jet_count)
		tile_jet_count[itile] = Int[]
	end

	# Find out where each jet lives, then push its index value to the correct tile
	@inbounds for ijet in 1:flat_jets._size
		ieta, iphi = get_tile(tiling_setup, eta(flat_jets, ijet), phi(flat_jets, ijet))
		push!(tile_jet_count[ieta, iphi], index(flat_jets, ijet))
	end

	# Now use the cached indexes to assign and fill the tiles
	@inbounds for itile in eachindex(tile_jet_count)
		ijets = tile_jet_count[itile]
		this_tile_jets = TiledJetSoA(length(ijets))
		@inbounds for (itilejet, ijet) in enumerate(ijets)
			set_kt2!(this_tile_jets, itilejet, kt2(flat_jets, ijet))
			set_eta!(this_tile_jets, itilejet, eta(flat_jets, ijet))
			set_phi!(this_tile_jets, itilejet, phi(flat_jets, ijet))
			set_index!(this_tile_jets, itilejet, index(flat_jets, ijet))
			set_nn!(this_tile_jets, itilejet, TiledSoACoord(0,0))
			set_nndist!(this_tile_jets, itilejet, R2)
		end
		tile_jets[itile] = this_tile_jets
	end
	populate_tile_cache!(tile_jets, tiling_setup)
end

"""
For each tile, populate a cache of the nearest tile neighbours
"""
function populate_tile_cache!(tile_jets::Array{TiledJetSoA, 2}, tiling_setup::TilingDef)
	# To help with later iterations, we now find and cache neighbour tile indexes
	@inbounds for ieta in 1:tiling_setup._n_tiles_eta
		@inbounds for iphi in 1:tiling_setup._n_tiles_phi
			# Clamping ensures we don't go beyond the limits of the eta tiling (which do not wrap)
			@inbounds for jeta in clamp(ieta - 1, 1, tiling_setup._n_tiles_eta):clamp(ieta + 1, 1, tiling_setup._n_tiles_eta)
				δeta = jeta - ieta
				@inbounds for jphi in iphi-1:iphi+1
					if (jeta == ieta && jphi == iphi)
						continue
					end
					# Phi tiles wrap around to meet each other
					δphi = jphi - iphi # Hold this unwrapped value for rightmost comparison
					if (jphi == 0)
						jphi = tiling_setup._n_tiles_phi
					elseif (jphi == tiling_setup._n_tiles_phi + 1)
						jphi = 1
					end
					# Tile is a neighbour
					tile_index = tiling_setup._tile_linear_indexes[jeta, jphi]
					push!(tile_jets[ieta, iphi]._nntiles, tile_index)
					# Only the tile directly above or to the right are _righttiles
					if (((δeta == -1) && (δphi == 0)) || (δphi == 1))
						push!(tile_jets[ieta, iphi]._righttiles, tile_index)
					end
				end
			end
		end
	end
end
