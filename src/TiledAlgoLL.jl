# Implementation of Tiled Algorithm, using linked list
# This is very similar to FastJet's N2Tiled algorithm

using Logging

# TODO: Consider ENUM here, rather than magic numbers
const Invalid=-3
const NonexistentParent=-2
const BeamJet=-1

"""A struct holding a record of jet mergers and finalisations"""
struct HistoryElement
    """Index in _history where first parent of this jet
    was created (NonexistentParent if this jet is an
    original particle)"""
    parent1::Int

    """index in _history where second parent of this jet
    was created (NonexistentParent if this jet is an
    original particle); BeamJet if this history entry
    just labels the fact that the jet has recombined
    with the beam)"""
    parent2::Int

    """index in _history where the current jet is
    recombined with another jet to form its child. It
    is Invalid if this jet does not further
    recombine."""
    child::Int

    """index in the _jets vector where we will find the
    PseudoJet object corresponding to this jet
    (i.e. the jet created at this entry of the
    history). NB: if this element of the history
    corresponds to a beam recombination, then
    jetp_index=Invalid."""
    jetp_index::Int

    """the distance corresponding to the recombination
       at this stage of the clustering."""
    dij::Float64

    """the largest recombination distance seen
       so far in the clustering history."""
    max_dij_so_far::Float64
end

"""Used for initial particles"""
HistoryElement(jetp_index) = HistoryElement(NonexistentParent, NonexistentParent, Invalid, jetp_index, 0.0, 0.0)

"""
Initialise the clustering history in a standard way,
Takes as input the list of stable particles as input
Returns the history and the total event energy.
"""
function initial_history(particles)
    # reserve sufficient space for everything
    history = Vector{HistoryElement}(undef, length(particles))
    sizehint!(history, 2*length(particles))

    Qtot::Float64 = 0

    for i in eachindex(particles)
        history[i] = HistoryElement(i)

        # get cross-referencing right from PseudoJets
        particles[i]._cluster_hist_index = i

        # determine the total energy in the event
        Qtot += particles[i].E
    end
    history, Qtot
end

"""Structure analogous to BriefJet, but with the extra information
needed for dealing with tiles"""
mutable struct TiledJet
    id::Int
    eta::Float64
    phi::Float64
    kt2::Float64
    NN_dist::Float64

    jets_index::Int
    tile_index::Int
    diJ_posn::Int


    "Nearest neighbour"
    NN::TiledJet

    previous::TiledJet
    next::TiledJet
    TiledJet(::Type{Nothing}) = begin
        t = new(-1, 0., 0., 0., 0., -1, 0, 0)
        t.NN = t.previous = t.next = t
        t
    end
    
    TiledJet(id, eta, phi, kt2, NN_dist,
             jet_index, tile_index, diJ_posn,
             NN, previous, next) = new(id, eta, phi, kt2, NN_dist,
                                       jet_index, tile_index, diJ_posn,
                                       NN, previous, next)
end


const noTiledJet::TiledJet = TiledJet(Nothing)

isvalid(t::TiledJet) = !(t === noTiledJet)

"""
Move a TiledJet in front of a TiledJet list element
The jet to move can be an isolated jet, a jet from another list or a jet from the same list
"""
insert!(nextjet::TiledJet, jettomove::TiledJet) = begin
    if !isnothing(nextjet)
        nextjet.previous  = jettomove
    end

    jettomove.next = nextjet
    jettomove.previous = nextjet.previous
    nextjet = jettomove
end

"""Detach a TiledJet from its list"""
detach!(jet::TiledJet) = begin
    if !isnothing(jet.previous)
        jet.previous.next = jet.next
    end
    if !isnothing(jet.next)
        jet.next.previous = jet.previous
    end
    jet.next = jet.previous = noTiledJet
end

TiledJet(id) = TiledJet(id, 0., 0., 0., 0.,
                        0, 0, 0,
                        noTiledJet, noTiledJet, noTiledJet)

import Base.copy
copy(j::TiledJet) = TiledJet(j.id, j.eta, j.phi, j.kt2, j.NN_dist, j.jets_index, j.tile_index, j.diJ_posn, j.NN, j.previous, j.next)

Base.iterate(tj::TiledJet) = (tj, tj)
Base.iterate(tj::TiledJet, state::TiledJet) = begin
    isvalid(state.next) ? (state.next::TiledJet, state.next::TiledJet) : nothing
end

"""Computes distance in the (eta,phi)-plane
between two jets."""
_tj_dist(jetA, jetB) = begin
    dphi = π - abs(π - abs(jetA.phi - jetB.phi))
    deta = jetA.eta - jetB.eta
    return dphi*dphi + deta*deta
end

_tj_diJ(jet) = begin
    kt2 = jet.kt2
    if isvalid(jet.NN) && jet.NN.kt2 < kt2
        kt2 = jet.NN.kt2
    end
    return jet.NN_dist * kt2
end

"""
Structure with the tiling parameters, as well as some bookkeeping
variables used during reconstruction
"""
struct Tiling
    setup::TilingDef
    tiles::Matrix{TiledJet}
    positions::Matrix{Int}
    tags::Matrix{Bool}
end

const tile_left = -1
const tile_central = 0
const tile_right = 1

const _n_tile_center = 1
const _n_tile_left_neighbours = 4
const _tile_right_neigbour_indices = 6:9
const _n_tile_right_neighbours = 4
const _n_tile_neighbours = 9

const neigh_init = fill(nothing, _n_tile_neighbours)

struct Surrounding{N}
    indices::NTuple{N, Int}
end

import Base.iterate

Base.iterate(x::T) where {T<:Surrounding} = (x.indices[1], 2)
Base.iterate(x::Surrounding{0}) = nothing
Base.iterate(x::Surrounding{1}, state) = nothing
Base.iterate(x::Surrounding{2}, state) = nothing
Base.iterate(x::Surrounding{3}, state) = state > 3 ? nothing : (x.indices[state], state+1)
Base.iterate(x::Surrounding{4}, state) = state > 4 ? nothing : (x.indices[state], state+1)
Base.iterate(x::Surrounding{6}, state) = state > 6 ? nothing : (x.indices[state], state+1)
Base.iterate(x::Surrounding{9}, state) = state > 9 ? nothing : (x.indices[state], state+1)

import Base.length
length(x::Surrounding{N}) where N = N

surrounding(center::Int, tiling::Tiling) = begin
    #                        4|6|9
    #                        3|1|8
    #                        2|5|7
    #  -> η

    iphip = mod1(center + tiling.setup._n_tiles_eta, tiling.setup._n_tiles)
    iphim = mod1(center - tiling.setup._n_tiles_eta, tiling.setup._n_tiles)

    if tiling.setup._n_tiles_eta == 1
        return Surrounding{3}((center, iphim, iphip))
    elseif tiling.positions[center] == tile_right
        return Surrounding{6}((center, iphim, iphip, iphim - 1, center - 1, iphip - 1))
    elseif tiling.positions[center] == tile_central
        return Surrounding{9}((center, iphim - 1, center - 1, iphip - 1,
                               iphim, iphip,
                               iphim + 1, center + 1, iphip + 1))
    else #tile_left
        return Surrounding{6}((center, iphim, iphip,
                               iphim + 1, center + 1, iphip + 1))
    end
end

rightneighbours(center::Int, tiling::Tiling) = begin
    #                         |1|4
    #                         | |3
    #                         | |2
    #  -> η

    iphip = mod1(center + tiling.setup._n_tiles_eta, tiling.setup._n_tiles)
    iphim = mod1(center - tiling.setup._n_tiles_eta, tiling.setup._n_tiles)

    if tiling.positions[center] == tile_right
        return Surrounding{1}((iphip,))
    else
        return Surrounding{4}((iphip, iphim + 1, center + 1, iphip + 1))
    end
end

"""Return a tiling setup with bookkeeping"""
Tiling(setup::TilingDef) = begin
    t = Tiling(setup,
               fill(noTiledJet, (setup._n_tiles_eta, setup._n_tiles_phi)),
               fill(0, (setup._n_tiles_eta, setup._n_tiles_phi)),
               fill(false, (setup._n_tiles_eta, setup._n_tiles_phi)))
    @inbounds for iphi = 1:setup._n_tiles_phi
        # The order of the following two statements is important
        # to have position = tile_right in case n_tiles_eta = 1
        t.positions[1, iphi] = tile_left
        t.positions[setup._n_tiles_eta, iphi] = tile_right
    end
    t
end

"""Return the tile index corresponding to the given eta,phi point"""
tile_index(tiling_setup, eta::Float64, phi::Float64) = begin
    if eta <= tiling_setup._tiles_eta_min
        ieta = 1
    elseif eta >= tiling_setup._tiles_eta_max
        ieta = tiling_setup._n_tiles_eta
    else
        ieta = 1 + unsafe_trunc(Int, (eta - tiling_setup._tiles_eta_min) / tiling_setup._tile_size_eta)
        # following needed in case of rare but nasty rounding errors
        if ieta > tiling_setup._n_tiles_eta
            ieta = tiling_setup._n_tiles_eta
        end
    end
    iphi = min(unsafe_trunc(Int, phi  / tiling_setup._tile_size_phi), tiling_setup._n_tiles_phi)
    return iphi * tiling_setup._n_tiles_eta + ieta
end

"""
Convienence structure holding all of the relevant parameters for
the jet reconstruction
"""
struct ClusterSequence
    """
    This contains the physical PseudoJets; for each PseudoJet one can find
    the corresponding position in the _history by looking at
    _jets[i].cluster_hist_index()
    """
    jets::Vector{PseudoJet}

    """
    This vector will contain the branching history; for each stage,
    history[i].jetp_index indicates where to look in the _jets
    vector to get the physical PseudoJet.
    """
    history::Vector{HistoryElement}

    """PseudoJet tiling"""
    tiling::Tiling

    """Total energy of the event"""
    Qtot
end


"""Initialise a tiled jet from a PseudoJet (using an index into our ClusterSequence)"""
tiledjet_set_jetinfo!(jet::TiledJet, cs::ClusterSequence, jets_index, R2) = begin
    @inbounds jet.eta  = rap(cs.jets[jets_index])
    @inbounds jet.phi  = phi_02pi(cs.jets[jets_index])
    @inbounds jet.kt2  = pt2(cs.jets[jets_index]) > 1.e-300 ? 1. / pt2(cs.jets[jets_index]) : 1.e300
    jet.jets_index = jets_index
    # Initialise NN info as well
    jet.NN_dist = R2
    jet.NN      = noTiledJet

    # Find out which tile it belonds to
    jet.tile_index = tile_index(cs.tiling.setup, jet.eta, jet.phi)

    # Insert it into the tile's linked list of jets (at the beginning)
    jet.previous = noTiledJet
    @inbounds jet.next = cs.tiling.tiles[jet.tile_index]
    if isvalid(jet.next) jet.next.previous = jet; end
    @inbounds cs.tiling.tiles[jet.tile_index] = jet
    nothing
end

"""Full scan for nearest neighbours"""
function set_nearest_neighbours!(cs::ClusterSequence, tiledjets::Vector{TiledJet})
    # Setup the initial nearest neighbour information
    for tile in cs.tiling.tiles
        isvalid(tile) || continue
        for jetA in tile
            for jetB in tile
                if jetB == jetA break; end
                dist = _tj_dist(jetA, jetB)
                if (dist < jetA.NN_dist)
                    jetA.NN_dist = dist
                    jetA.NN = jetB
                end
                if dist < jetB.NN_dist
                    jetB.NN_dist = dist
                    jetB.NN = jetA
                end
            end
        end

        # Look for neighbour jets n the neighbour tiles
        for rtile_index in rightneighbours(tile.tile_index, cs.tiling)
            for jetA in tile
                for jetB in @inbounds cs.tiling.tiles[rtile_index]
                    dist = _tj_dist(jetA, jetB)
                    if (dist < jetA.NN_dist)
                        jetA.NN_dist = dist
                        jetA.NN = jetB
                    end
                    if dist < jetB.NN_dist
                        jetB.NN_dist = dist
                        jetB.NN = jetA
                    end
                end
            end
            # No need to do it for LH tiles, since they are implicitly done
            # when we set NN for both jetA and jetB on the RH tiles.
        end
    end

    # Now create the diJ (where J is i's NN) table - remember that
    # we differ from standard normalisation here by a factor of R2
    # (corrected for at the end).
    diJ = similar(cs.jets, Float64)
    NNs = similar(cs.jets, TiledJet)
    for i in eachindex(diJ)
        jetA = tiledjets[i]
        diJ[i] = _tj_diJ(jetA) # kt distance * R^2
        # our compact diJ table will not be in one-to-one corresp. with non-compact jets,
        # so set up bi-directional correspondence here.
        @inbounds NNs[i] = jetA  
        jetA.diJ_posn = i
    end
end

"""
Jet reconstruction algorithm
"""
function tiled_jet_reconstruct_ll(objects::AbstractArray{T}; p = -1, R = 1.0, recombine = +) where T
    # Bounds
	N::Int = length(objects)
	@debug "Initial particles: $(N)"

	# Algorithm parameters
	R2::Float64 = R * R
    invR2::Float64 = 1/R2
	p = (round(p) == p) ? Int(p) : p # integer p if possible

    # Container for pseudojets, sized for all initial particles, plus all of the
    # merged jets that can be created during reconstruction
    jets = PseudoJet[]
    sizehint!(jets, N*2)
    resize!(jets, N)

    # Copy input data into the jets container
    # N.B. Could specialise to accept PseudoJet objects directly (which is what HepMC3.jl reader provides)
    for i in 1:N
        jets[i] = PseudoJet(px(objects[i]), py(objects[i]), pz(objects[i]), energy(objects[i]))
    end

    # Setup the initial history and get the total energy
    history, Qtot = initial_history(jets)

    # Now get the tiling setup
    _eta = JetReconstruction.eta.(objects) # This could be avoided, probably...
    tiling_setup = Tiling(setup_tiling(_eta, R))

    # ClusterSequence is a convenience struct that holds the state of the reconstruction
    clusterseq = ClusterSequence(jets, history, tiling_setup, Qtot)

    # Tiled jets is a structure that has additional variables for tracking which tile a jet is in
    tiledjets = similar(clusterseq.jets, TiledJet)
    for ijet in eachindex(tiledjets)
        tiledjets[ijet] = TiledJet(ijet)
        tiledjet_set_jetinfo!(tiledjets[ijet], clusterseq, ijet, R2)
    end

    # Now initalise all of the nearest neighbour tiles
    set_nearest_neighbours!(clusterseq, tiledjets)

    # Implement me please...
    return T[], T[]
end
