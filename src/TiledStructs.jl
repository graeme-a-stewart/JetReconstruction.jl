# Defined the structures and associated functions used in tiled
# jet reconstruction

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

"""Nearest neighbour coordinates"""
mutable struct TiledNN{I}
    _itile::I           # Jet tile index (flattened)
    _ijet::I            # Jet position in this tile
end

"""Setter for nearest neighbour"""
set_nn!(mynn::TiledNN, itile, ijet) = begin
    mynn._itile = itile
    mynn._ijet = ijet
end

"""Do we have a NN, or not"""
valid_nn(mynn::TiledNN) = begin
    return mynn._itile > 0
end

"""Structure of arrays for tiled jet parameters, using an SoA layout
for computational efficiency"""
mutable struct TiledJetSoA{F, I}
    _size::I                # Active jet count (can be less than the vector length)
	_kt2::Vector{F}         # p_t^-2p
	_eta::Vector{F}         # Rapidity
	_phi::Vector{F}         # Phi coordinate
	_index::Vector{I}       # My jet index
	_nn::Vector{TiledNN{I}} # Nearest neighbour location (if (0,0) no nearest neighbour)
	_nndist::Vector{F}      # Distance to my nearest neighbour
    _dij::Vector{F}         # Jet metric distance to my nearest neighbour
    _righttiles::Vector{I}  # Indexes of all tiles to my right
    _nntiles::Vector{I}     # Indexes of all neighbour tiles
end

"""Return the NN index of a nearest neighbour tile"""
nnindex(tile_jets::Array{TiledJetSoA, 2}, itile, ijet) = begin
    return tile_jets[tile_jets[itile]._nn[ijet]._itile]._index[tile_jets[itile]._nn[ijet]._ijet]
end

"""Constructor for a tile holding n jets"""
TiledJetSoA{F, I}(n::Integer) where {F, I} = TiledJetSoA{F, I}(
    n,
	Vector{F}(undef, n),
	Vector{F}(undef, n),
	Vector{F}(undef, n),
	Vector{I}(undef, n),
	Vector{TiledNN}(undef, n),
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

"""Return the nth jet in the SoA"""
get_jet(j, n::Integer) = begin
    return j._index[n], j._eta[n], j._phi[n], j._kt2[n]
end

# Accessors - will work for both SoAs
kt2(j, n::Int) = j._kt2[n]
eta(j, n::Int) = j._eta[n]
phi(j, n::Int) = j._phi[n]
index(j, n::Int) = j._index[n]
nn(j, n::Int) = j._nn[n]
nndist(j, n::Int) = j._nndist[n]
