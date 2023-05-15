### Tiled Jet Reconstruction 

struct TilingDef{F,I}
    _tiles_eta_min::F
    _tiles_eta_max::F
    _tile_size_eta::F
    _tile_size_phi::F
    _n_tiles_eta::I
    _n_tiles_phi::I
    _n_tiles::I
    _tiles_ieta_min::I
    _tiles_ieta_max::I
end

"""

"""
function _determine_rapidity_extent(_eta::Vector{T}) where T<:AbstractFloat
    """"Have a binning of rapidity that goes from -nrap to nrap
    in bins of size 1; the left and right-most bins include
    include overflows from smaller/larger rapidities"""
    
    length(_eta) == 0 && return 0., 0.

    const nrap = 20
    const nbins = 2*nrap
    counts = zeros(Int, nbins)

    # Get the minimum and maximum rapidities and at the same time bin
    # the multiplicities as a function of rapidity to help decide how
    # far out it's worth going
    minrap =  floatmax(T)
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

    # start scan over rapidity bins from the left, to find out minimum rapidity of tiling
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

    # then do it from right, to find out maximum rapidity of tiling
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

    println("$(minrap) - $(maxrap)")
    exit(0)

    minrap, maxrap
end

function _initial_tiling(particles, Rparam)

    # first decide tile sizes (with a lower bound to avoid huge memory use with
    # very small R)
    tile_size_eta = max(0.1, Rparam)

    # it makes no sense to go below 3 tiles in phi -- 3 tiles is
    # sufficient to make sure all pair-wise combinations up to pi in
    # phi are possible
    n_tiles_phi   = max(3, floor(Int, 2π/tile_size_eta))

    tile_size_phi = 2π / n_tiles_phi # >= Rparam and fits in 2pi

    tiles_eta_min, tiles_eta_max = determine_rapidity_extent(particles)

    # now adjust the values
    tiles_ieta_min = floor(Int, tiles_eta_min/tile_size_eta)
    tiles_ieta_max = floor(Int, tiles_eta_max/tile_size_eta) #FIXME shouldn't it be ceil ?
    tiles_eta_min = tiles_ieta_min * tile_size_eta
    tiles_eta_max = tiles_ieta_max * tile_size_eta
    n_tiles_eta = tiles_ieta_max - tiles_ieta_min + 1

    tiling_setup = TilingDef(tiles_eta_min, tiles_eta_max,
                             tile_size_eta, tile_size_phi,
                             n_tiles_eta, n_tiles_phi, n_tiles_eta*n_tiles_phi,
                             tiles_ieta_min, tiles_ieta_max)

    # allocate the tiles
    tiling_size = (n_tiles_eta, n_tiles_phi)
    Tiling(tiling_setup)
end

"""
Tiled jet reconstruction
"""
function tiled_jet_reconstruct(objects::AbstractArray{T}; p=-1, R=1.0, recombine=+) where T
    # bounds
    N::Int = length(objects)

    # returned values
    jets = T[] # result
    sequences = Vector{Int}[] # recombination sequences, WARNING: first index in the sequence is not necessarily the seed

    # params
    _R2::Float64 = R*R
    _p = (round(p) == p) ? Int(p) : p # integer p if possible
    ap = abs(_p); # absolute p

    # data
    _objects = copy(objects)
    _kt2 = (JetReconstruction.pt.(_objects) .^ 2) .^ _p
    _phi = JetReconstruction.phi.(_objects)
    _eta = JetReconstruction.eta.(_objects)
    _nn = Vector(1:N) # nearest neighbours
    _nndist = fill(float(_R2), N) # distances to the nearest neighbour
    _sequences = Vector{Int}[[x] for x in 1:N]

    # tiling
    _determine_rapidity_extent(_eta)

end