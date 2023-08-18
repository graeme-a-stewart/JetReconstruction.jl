# Implementation of Tiled Algorithm, using linked list
# This is very similar to FastJet's N2Tiled algorithm

using Logging
using Accessors

# Include struct definitions and basic operations
include("TiledAlglLLStructs.jl")

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


"""Initialise a tiled jet from a PseudoJet (using an index into our ClusterSequence)"""
tiledjet_set_jetinfo!(jet::TiledJet, clusterseq::ClusterSequence, jets_index, R2) = begin
    @inbounds jet.eta  = rap(clusterseq.jets[jets_index])
    @inbounds jet.phi  = phi_02pi(clusterseq.jets[jets_index])
    @inbounds jet.kt2  = pt2(clusterseq.jets[jets_index]) > 1.e-300 ? 1. / pt2(clusterseq.jets[jets_index]) : 1.e300
    jet.jets_index = jets_index
    # Initialise NN info as well
    jet.NN_dist = R2
    jet.NN      = noTiledJet

    # Find out which tile it belonds to
    jet.tile_index = tile_index(clusterseq.tiling.setup, jet.eta, jet.phi)

    # Insert it into the tile's linked list of jets (at the beginning)
    jet.previous = noTiledJet
    @inbounds jet.next = clusterseq.tiling.tiles[jet.tile_index]
    if isvalid(jet.next) jet.next.previous = jet; end
    @inbounds clusterseq.tiling.tiles[jet.tile_index] = jet
    nothing
end


"""Full scan for nearest neighbours"""
function set_nearest_neighbours!(clusterseq::ClusterSequence, tiledjets::Vector{TiledJet})
    # Setup the initial nearest neighbour information
    for tile in clusterseq.tiling.tiles
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
        for rtile_index in rightneighbours(tile.tile_index, clusterseq.tiling)
            for jetA in tile
                for jetB in @inbounds clusterseq.tiling.tiles[rtile_index]
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
    diJ = similar(clusterseq.jets, Float64)
    NNs = similar(clusterseq.jets, TiledJet)
    for i in eachindex(diJ)
        jetA = tiledjets[i]
        diJ[i] = _tj_diJ(jetA) # kt distance * R^2
        # our compact diJ table will not be in one-to-one corresp. with non-compact jets,
        # so set up bi-directional correspondence here.
        @inbounds NNs[i] = jetA  
        jetA.diJ_posn = i
    end
    NNs, diJ
end

"""Carries out the bookkeeping associated with the step of recombining
jet_i and jet_j (assuming a distance dij) and returns the index
of the recombined jet, newjet_k."""
do_ij_recombination_step!(clusterseq::ClusterSequence, jet_i, jet_j, dij) = begin
    # Create the new jet by recombining the first two with
    # the E-scheme
    push!(clusterseq.jets, clusterseq.jets[jet_i] + clusterseq.jets[jet_j])

    # Get its index and the history index
    newjet_k = length(clusterseq.jets)
    newstep_k = length(clusterseq.history) + 1

    # And provide jet with this info
    clusterseq.jets[newjet_k]._cluster_hist_index = newstep_k

    # Finally sort out the history
    hist_i = clusterseq.jets[jet_i]._cluster_hist_index
    hist_j = clusterseq.jets[jet_j]._cluster_hist_index

    add_step_to_history!(clusterseq, minmax(hist_i, hist_j)...,
                         newjet_k, dij)

    newjet_k
end

"""Carries out the bookkeeping associated with the step of recombining
jet_i with the beam (i.e., finalising the jet)"""
do_iB_recombination_step!(clusterseq::ClusterSequence, jet_i, diB) = begin
    # Recombine the jet with the beam
    add_step_to_history!(clusterseq, clusterseq.jets[jet_i]._cluster_hist_index, BeamJet,
                         Invalid, diB)
end

"""Add a new jet's history into the recombination sequence"""
add_step_to_history!(clusterseq::ClusterSequence, parent1, parent2, jetp_index, dij) = begin
    max_dij_so_far = max(dij, clusterseq.history[end].max_dij_so_far)
    push!(clusterseq.history, HistoryElement(parent1, parent2, Invalid,
                                     jetp_index, dij, max_dij_so_far))

    local_step = length(clusterseq.history)

    # Sanity check: make sure the particles have not already been recombined
    #
    # Note that good practice would make this an assert (since this is
    # a serious internal issue). However, we decided to throw an
    # InternalError so that the end user can decide to catch it and
    # retry the clustering with a different strategy.

    @assert parent1 >= 1
    if clusterseq.history[parent1].child != Invalid
        throw(ErrorException("Internal error. Trying to recombine an object that has previsously been recombined."))
    end

    hist_elem = clusterseq.history[parent1]
    clusterseq.history[parent1] = @set hist_elem.child = local_step

    if parent2 >= 1
        clusterseq.history[parent2].child == Invalid || error("Internal error. Trying to recombine an object that has previsously been recombined.  Parent " * string(parent2) * "'s child index " * string(clusterseq.history[parent1].child) * ". Parent jet index: " * string(clusterseq.history[parent2].jetp_index) * ".")
        hist_elem = clusterseq.history[parent2]
        clusterseq.history[parent2] = @set hist_elem.child = local_step
    end

    # Get cross-referencing right from PseudoJets
    if jetp_index != Invalid
        @assert jetp_index >= 1
        clusterseq.jets[jetp_index]._cluster_hist_index = local_step
    end
end


"""Find the lowest value in the array, returning the value and the index"""
find_lowest(diJ, n) = begin
    best = 1
    @inbounds diJ_min = diJ[1]
    @turbo for here in 2:n
        newmin = diJ[here] < diJ_min
        best = newmin ? here : best
        diJ_min = newmin ? diJ[here] : diJ_min
    end
    # @assert argmin(diJ[1:n]) == best
    diJ_min, best
end

"""
Main jet reconstruction algorithm
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
    NNs, dij = set_nearest_neighbours!(clusterseq, tiledjets)

    # Main loop of the reconstruction
    # Each iteration we either merge 2→1 or finalise a jet, so it takes N iterations
    # to complete the reconstruction

    for iteration in 1:N
        # Search for the lowest value of min_dij_ijet
        dij_min, ibest = find_lowest(dij, N - (iteration-1))
        next_history_location = length(clusterseq.jets)+1
        @inbounds jetA = NNs[ibest]
        jetB = jetA.NN

        # Normalisation
        dij_min *= R2

        @debug "Iteration $(iteration): dij_min $(dij_min); jetA $(jetA.id), jetB $(jetB.id)"

        if isvalid(jetB)
            # Jet-jet recombination
            # If necessary relabel A & B to ensure jetB < jetA, that way if
            # the larger of them == newtail then that ends up being jetA and
            # the new jet that is added as jetB is inserted in a position that
            # has a future!
            if jetA.id < jetB.id
                jetA, jetB = jetB, jetA;
            end

            # Recombine jetA and jetB and retrieves the new index, nn
            nn = do_ij_recombination_step!(clusterseq, jetA.jets_index, jetB.jets_index, dij_min)
            tiledjet_remove_from_tiles!(clusterseq.tiling, jetA)
            oldB = copy(jetB)  # take a copy because we will need it...

            tiledjet_remove_from_tiles!(clusterseq.tiling, jetB)
            tiledjet_set_jetinfo!(jetB, clusterseq, nn, R2) # cause jetB to become _jets[nn]
            #                                  (in addition, registers the jet in the tiling)
        else
            # Jet-beam recombination
            do_iB_recombination_step!(clusterseq, jetA.jets_index, dij_min)
            tiledjet_remove_from_tiles!(clusterseq.tiling, jetA)
        end

        exit(0)

    end

    # Implement me please...
    return T[], T[]
end
