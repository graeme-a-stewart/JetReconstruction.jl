### Tiled Jet Reconstruction 

"""Tiling definition parameters"""
struct TilingDef{F, I}
	_tiles_eta_min::F   # Minimum rapidity
	_tiles_eta_max::F   # Maximum rapidity
	_tile_size_eta::F   # Size of a tile in rapidity (usually R^2)
	_tile_size_phi::F   # Size of a tile in phi (usually a bit more than R^2)
	_n_tiles_eta::I     # Number of tiles across rapidity
	_n_tiles_phi::I     # Number of tiles across phi
	_n_tiles::I         # Total number of tiles
	_tiles_ieta_min::I  # Min_rapidity / rapidity tile size (needed?)
	_tiles_ieta_max::I  # Max_rapidity / rapidity tile size (needed?)
end

"""Structure of arrays for tiled jet parameters"""
mutable struct TiledJetSoA{F, I}
    _size::I                # Active jet count (can be less than the vector length)
	_kt2::Vector{F}         # p_t^-2
	_eta::Vector{F}         # Rapidity
	_phi::Vector{F}         # Phi coordinate
	_index::Vector{I}       # My jet index
	_nn::Vector{I}          # Nearest neighbour index (if 0, no nearest neighbour)
	_nndist::Vector{F}      # Distance to my nearest neighbour
    _dij::Vector{F}         # Jet metric distance to my nearest neighbour
    _righttiles::Vector{I}  # Indexes of all tiles to my right
    _nntiles::Vector{I}     # Indexes of all neighbour tiles
end

TiledJetSoA{F, I}(n::Integer) where {F, I} = TiledJetSoA{F, I}(
    n,
	Vector{F}(undef, n),
	Vector{F}(undef, n),
	Vector{F}(undef, n),
	Vector{I}(undef, n),
	Vector{I}(undef, n),
	Vector{F}(undef, n),
    Vector{F}(undef, n),
    Vector{I}(undef, 0),
    Vector{I}(undef, 0)
)

"""Structure for the flat jet SoA, as it's convenient"""
mutable struct FlatJetSoA{F, I}
	_size::Int            # Number of active entries (may be less than the vector size!)
	_kt2::Vector{F}       # p_t^-2
	_eta::Vector{F}       # Rapidity
	_phi::Vector{F}       # Phi coordinate
	_index::Vector{I}     # My jet index
	_nn::Vector{I}        # Nearest neighbour index (if 0, no nearest neighbour)
	_nndist::Vector{F}    # Geometric distance to my nearest neighbour
    _dij::Vector{F}       # Jet metric distance to my nearest neighbour
end

# Accessors - will work for both SoAs
kt2(j, n::Int) = j._kt2[n]
eta(j, n::Int) = j._eta[n]
phi(j, n::Int) = j._phi[n]
index(j, n::Int) = j._index[n]
nn(j, n::Int) = j._nn[n]
nndist(j, n::Int) = j._nndist[n]

"""
Determine an extent for the rapidity tiles, by binning in 
rapidity and collapsing the outer bins until they have about
1/4 the number of particles as the maximum bin. This is the 
heuristic which is used by FastJet.
"""
function _determine_rapidity_extent(_eta::Vector{T}) where T <: AbstractFloat
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

function _setup_tiling(_eta::Vector{T}, Rparam::AbstractFloat) where T <: AbstractFloat
	# First decide tile sizes (with a lower bound to avoid huge memory use with
	# very small R)
	tile_size_eta = max(0.1, Rparam)

	# It makes no sense to go below 3 tiles in phi -- 3 tiles is
	# sufficient to make sure all pair-wise combinations up to pi in
	# phi are possible
	n_tiles_phi   = max(3, floor(Int, 2π / tile_size_eta))
	tile_size_phi = 2π / n_tiles_phi # >= Rparam and fits in 2pi

	tiles_eta_min, tiles_eta_max = _determine_rapidity_extent(_eta)

	# now adjust the values
	tiles_ieta_min = floor(Int, tiles_eta_min / tile_size_eta)
	tiles_ieta_max = floor(Int, tiles_eta_max / tile_size_eta) #FIXME shouldn't it be ceil ?
	tiles_eta_min = tiles_ieta_min * tile_size_eta
	tiles_eta_max = tiles_ieta_max * tile_size_eta
	n_tiles_eta = tiles_ieta_max - tiles_ieta_min + 1

	tiling_setup = TilingDef(tiles_eta_min, tiles_eta_max,
		tile_size_eta, tile_size_phi,
		n_tiles_eta, n_tiles_phi, n_tiles_eta * n_tiles_phi,
		tiles_ieta_min, tiles_ieta_max)

	println(tiling_setup)
	# exit(0)

	tile_jets = Array{TiledJetSoA, 2}(undef, n_tiles_eta, n_tiles_phi)
	tiling_setup, tile_jets

	# allocate the tiles
	# tiling_size = (n_tiles_eta, n_tiles_phi)
	# Tiling(tiling_setup)
end

"""
Return the geometric distance between a pair of (eta,phi) coordinates
"""
geometric_distance(eta1::AbstractFloat, phi1::AbstractFloat, eta2::AbstractFloat, phi2::AbstractFloat) = begin
    δeta = eta2-eta1
    δphi = π - abs(π - abs(phi1 - phi2))
    return δeta*δeta + δphi*δphi
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
Populate tiling structure with our initial jets and setup neighbour tile caches
"""
function populate_tiles!(tile_jets::Array{TiledJetSoA, 2}, tiling_setup::TilingDef, flat_jets::FlatJetSoA)
	# This is a special case, where the initial particles are all
	# "linear" in the flat_jets structure, so we scan through that
	# and match each jet to a tile, so that we can assign correct size
	# vectors in the tiled jets structure
	tile_jet_count = Array{Vector{Int}, 2}(undef, tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi)
	# Using fill() doesn't work as we fill all tiles with the same vector!
	for itile in eachindex(tile_jet_count)
		tile_jet_count[itile] = Int[]
	end
	
    # Find out where each jet lives, then push its index value to the correct tile
    for ijet in 1:flat_jets._size
		ieta, iphi = get_tile(tiling_setup, eta(flat_jets, ijet), phi(flat_jets, ijet))
		push!(tile_jet_count[ieta, iphi], index(flat_jets, ijet))
	end

    # Now use the cached indexes to assign and fill the tiles
    for itile in eachindex(tile_jet_count)
        ijets = tile_jet_count[itile]
        this_tile_jets = TiledJetSoA{Float64, Int}(length(ijets))
        for (itilejet, ijet) in enumerate(ijets)
            this_tile_jets._kt2[itilejet] = flat_jets._kt2[ijet]
            this_tile_jets._eta[itilejet] = flat_jets._eta[ijet]
            this_tile_jets._phi[itilejet] = flat_jets._phi[ijet]
            this_tile_jets._index[itilejet] = flat_jets._index[ijet]
            this_tile_jets._nn[itilejet] = flat_jets._nn[ijet]
            this_tile_jets._nndist[itilejet] = flat_jets._nndist[ijet]
        end
        tile_jets[itile] = this_tile_jets
        # println("$(itile) - $(this_tile_jets)")
    end
    populate_tile_cache!(tile_jets, tiling_setup)
end

"""
For each tile, populate a cache of the nearest tile neighbours
"""
function populate_tile_cache!(tile_jets::Array{TiledJetSoA, 2}, tiling_setup::TilingDef)
    # To help with later iterations, we now find and cache neighbour tile indexes
    tile_linear_indexes = LinearIndices(tile_jets)
    for ieta in 1:tiling_setup._n_tiles_eta
        for iphi in 1:tiling_setup._n_tiles_phi
            # Clamping ensures we don't go beyond the limits of the eta tiling (which do not wrap)
            for jeta in clamp(ieta-1,1,tiling_setup._n_tiles_eta):clamp(ieta+1,1,tiling_setup._n_tiles_eta)
                δeta = jeta-ieta
                for jphi in iphi-1:iphi+1
                    if (jeta==ieta && jphi==iphi) continue end
                    # Phi tiles wrap around to meet each other
                    δphi = jphi-iphi # Hold this unwrapped value for rightmost comparison
                    if (jphi==0)
                        jphi=tiling_setup._n_tiles_phi 
                    elseif (jphi==tiling_setup._n_tiles_phi+1)
                        jphi=1
                    end
                    # Tile is a neighbour
                    push!(tile_jets[ieta,iphi]._nntiles, tile_linear_indexes[jeta, jphi])
                    # Only the tile directly above or to the right are _righttiles
                    if (((δeta==-1) && (δphi==0)) || (δphi==1))
                        push!(tile_jets[ieta,iphi]._righttiles, tile_linear_indexes[jeta, jphi])
                    end
                end
            end
        end
    end
end


"""
Do a complete scan over tiles to final all nearest neighbour distances
"""
function find_all_nearest_neighbours!(tile_jets::Array{TiledJetSoA, 2}, tiling_setup::TilingDef, 
    flat_jets::FlatJetSoA, R2::AbstractFloat)
    # We march over all tiles, evaluating the distance to
    # - each jet in this tile
    # - each jet in the rightmost neighbour tiles
    # As we compare jet-to-jet, in both directions, the rightmost tiles ensure that the
    # whole space is swept
    for tile in tile_jets
        if (tile._size==0)
            continue
        end
        for ijet in 1:tile._size
            # Could we do this in a broadcast way...? Would it be faster?
            for jjet in ijet+1:tile._size
                _dist = geometric_distance(tile._eta[ijet], tile._phi[ijet],
                tile._eta[jjet], tile._phi[jjet])
                if (_dist < tile._nndist[ijet])
                    tile._nndist[ijet] = _dist
                    tile._nn[ijet] = tile._index[jjet]
                end
                if (_dist < tile._nndist[jjet])
                    tile._nndist[jjet] = _dist
                    tile._nn[jjet] = tile._index[ijet]
                end
            end
            for jtile in tile._righttiles
                tile2 = tile_jets[jtile]
                for jjet in 1:tile2._size
                    _dist = geometric_distance(tile._eta[ijet], tile._phi[ijet],
                    tile2._eta[jjet], tile2._phi[jjet])
                    if (_dist < tile._nndist[ijet])
                        tile._nndist[ijet] = _dist
                        tile._nn[ijet] = tile2._index[jjet]
                    end
                    if (_dist < tile2._nndist[jjet])
                        tile2._nndist[jjet] = _dist
                        tile2._nn[jjet] = tile._index[ijet]
                    end
                end
            end
        end
    end

    # Now calculate the dij distances
    min_dij = 1e20
    min_dij_itile = 0
    min_dij_ijet = 0
    for (itile, tile) in enumerate(tile_jets)
        for ijet in 1:tile._size
            if tile._nn[ijet] == 0
                tile._dij[ijet] = tile._nndist[ijet] * tile._kt2[ijet]
            else
                tile._dij[ijet] = tile._nndist[ijet] * min(tile._kt2[ijet], flat_jets._kt2[tile._nn[ijet]])
            end
            # And as this is a complete scan, find the minimum dij as well
            if tile._dij[ijet] < min_dij
                min_dij = tile._dij[ijet]
                min_dij_itile = itile
                min_dij_ijet = ijet
            end
        end
    end
    min_dij_itile, min_dij_ijet, min_dij
end

"""
Tiled jet reconstruction
"""
function tiled_jet_reconstruct(objects::AbstractArray{T}; p = -1, R = 1.0, recombine = +) where T
	# bounds
	N::Int = length(objects)

	# returned values
	jets = T[] # result
	sequences = Vector{Int}[] # recombination sequences, WARNING: first index in the sequence is not necessarily the seed

	# params
	_R2::Float64 = R * R
	_p = (round(p) == p) ? Int(p) : p # integer p if possible
	ap = abs(_p) # absolute p

	# data (make this a struct?)
	_objects = copy(objects)
	_kt2 = (JetReconstruction.pt.(_objects) .^ 2) .^ _p
	_phi = JetReconstruction.phi.(_objects)
	_eta = JetReconstruction.eta.(_objects)
	_index = collect(1:N) # Initial jets are just numbered 1:N
	_nn = fill(0, N) # Nearest neighbours (0 -> self)
	_nndist = fill(float(_R2), N) # Distances to the nearest neighbour (set to self-distance, initially)
    _dij = fill(0.0, N) # dij
	_sequences = Vector{Int}[[x] for x in 1:N]

	flat_jets = FlatJetSoA{typeof(_kt2[1]), typeof(_index[1])}(N, _kt2, _eta, _phi, _index, _nn, _nndist, _dij)

	# Tiling
	tiling_setup, tile_jets = _setup_tiling(_eta, R)

	# Populate initial tiles
	populate_tiles!(tile_jets, tiling_setup, flat_jets)
    # println(tile_jets)

    # Setup initial nn, nndist and dij values
    min_dij_itile, min_dij_ijet, min_dij = find_all_nearest_neighbours!(tile_jets, tiling_setup, flat_jets, _R2)
    println("$(min_dij) - $(min_dij_itile) - $(min_dij_ijet) $(tile_jets[min_dij_itile]._index[min_dij_ijet]) -> $(tile_jets[min_dij_itile]._nn[min_dij_ijet])")

	exit(0)
end
