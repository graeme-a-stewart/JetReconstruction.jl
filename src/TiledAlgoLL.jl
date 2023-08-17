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
    tiling_setup = setup_tiling(_eta, R)

    # Implement me please...
    return T[], T[]
end
