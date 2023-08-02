# Tiled jet reconstruction, linked list data structure approach

using Logging
using LoopVectorization

"""
Structure holding the flat jets for a tiled reconstruction
"""
struct FlatJets
    # Physics quantities
	kt2::Vector{Float64}       	# p_t^(-2*power)
	eta::Vector{Float64}        # Rapidity
	phi::Vector{Float64}        # Phi coordinate

    # Mapping to original jets in (px, py, pz, E)
    jet_index::Vector{Int}

    # Tiles and linked list 
	tile_index::Vector{Int}    # My tile index (this is the linear index)

    # Reconstruction parameters
    nearest_neighbour::Vector{Int}
    nn_distance::Vector{Float64}
    dij_distance::Vector{Float64}
end

# Accessor functions for the jet array
kt2(jets::FlatJets, n::Int) = jets.kt2[n]
eta(jets::FlatJets, n::Int) = jets.eta[n]
phi(jets::FlatJets, n::Int) = jets.phi[n]
jet_index(jets::FlatJets, n::Int) = jets.jet_index[n]
tile_index(jets::FlatJets, n::Int) = jets.tile_index[n]
nearest_neighbour(jets::FlatJets, n::Int) = jets.nearest_neighbour[n]
nn_distance(jets::FlatJets, n::Int) = jets.nn_distance[n]
dij_distance(jets::FlatJets, n::Int) = jets.dij_distance[n]

# Setters
set_kt2!(jets::FlatJets, n::Int, v) = jets.kt2[n] = v
set_eta!(jets::FlatJets, n::Int, v) = jets.eta[n] = v
set_phi!(jets::FlatJets, n::Int, v) = jets.phi[n] = v

set_jet_index!(jets::FlatJets, n::Int, i) = jets.jet_index[n] = i
set_tile_index!(jets::FlatJets, n::Int, i) = jets.tile_index[n] = i

set_nearest_neighbour!(jets::FlatJets, n::Int, v::Int) = jets.nearest_neighbour[n] = v
set_nn_distance!(jets::FlatJets, n::Int, v::Float64) = jets.nn_distance[n] = v
set_dij_distance!(jets::FlatJets, n::Int, v::Float64) = jets.dij_distance[n] = v

"""Add jet from (px, py, pz, E) values"""
function insert_flatjet!(jets::FlatJets, tiling_setup, p, n::Int, jet_index::Int, newjet::Vector{Float64})
    set_kt2!(jets, n, JetReconstruction.pt2(newjet) ^ p)
    set_eta!(jets, n, JetReconstruction.eta(newjet))
    set_phi!(jets, n, JetReconstruction.phi(newjet))
    set_jet_index!(jets, n, jet_index)
    tile_η, tile_ϕ = get_tile(tiling_setup, eta(jets, n), phi(jets, n))
    set_tile_index!(jets, n, get_tile_linear_index(tiling_setup, tile_η, tile_ϕ))
    set_nearest_neighbour!(jets, n, 0)
    set_nn_distance!(jets, n, 1e9)
    set_dij_distance!(jets, n, 1e9)
end

"""Suppress jet at index, copying in the last jet is needed"""
function suppress_flatjet!(jets::FlatJets, n::Int)
    # Is the jet we want to get rid of the final jet? In this case the job is trivial
    ilast = lastindex(jets.kt2)
    tainted_index::Int = 0
    if n != ilast
        # Not the last jet - need to shuffle...
        set_kt2!(jets, n, kt2(jets, ilast))
        set_eta!(jets, n, eta(jets, ilast))
        set_phi!(jets, n, phi(jets, ilast))
        set_jet_index!(jets, n, jet_index(jets, ilast))
        set_tile_index!(jets, n, tile_index(jets, ilast))
        set_nearest_neighbour!(jets, n, nearest_neighbour(jets, ilast))
        set_nn_distance!(jets, n, nn_distance(jets, ilast))
        set_dij_distance!(jets, n, dij_distance(jets, ilast))
        tainted_index = ilast
    end
    pop!(jets.kt2)
    pop!(jets.eta)
    pop!(jets.phi)
    pop!(jets.jet_index)
    pop!(jets.tile_index)
    pop!(jets.nearest_neighbour)
    pop!(jets.nn_distance)
    pop!(jets.dij_distance)
    # @debug ilast tainted_index n
    tainted_index
end


"""
Structure holding the tiles for the reconstruction
"""
struct Tile
    jets::Set{Int}
    function Tile()
        new(Set{Int}())
    end
end

"""
Populate the tiles with nearest neighbour information
"""
function populate_tile_nn!(tiles::Array{Tile, 2}, tiling_setup)
	# To help with later iterations, we now find and cache neighbour tile indexes
	@inbounds for iη in 1:tiling_setup._n_tiles_eta
		@inbounds for iϕ in 1:tiling_setup._n_tiles_phi
			# Clamping ensures we don't go beyond the limits of the eta tiling (which do not wrap)
			@inbounds for jη in clamp(iη - 1, 1, tiling_setup._n_tiles_eta):clamp(iη + 1, 1, tiling_setup._n_tiles_eta)
				δη = jη - iη
				@inbounds for jϕ in iϕ-1:iϕ+1
					if (jη == iη && jϕ == iϕ)
						continue
					end
					# Phi tiles wrap around to meet each other
					δϕ = jϕ - iϕ # Hold this unwrapped value for rightmost comparison
					if (jϕ == 0)
						jϕ = tiling_setup._n_tiles_phi
					elseif (jϕ == tiling_setup._n_tiles_phi + 1)
						jϕ = 1
					end
					# Tile is a neighbour
					tile_index = tiling_setup._tile_linear_indexes[jη, jϕ]
					push!(tiles[iη, iϕ].neighbour_tiles, tile_index)
					# Only the tile directly above or to the right are _righttiles
					if (((δη == -1) && (δϕ == 0)) || (δϕ == 1))
						push!(tiles[iη, iϕ].right_neighbour_tiles, tile_index)
					end
				end
			end
		end
	end
end

"""
Populate the tiles with their first jets and construct the linked lists in the flat jet array
"""
function populate_tile_lists!(tiles::Array{Tile, 2}, flatjets::FlatJets, tiling_setup::TilingDef)
    for ijet in 1:size(flatjets.eta)[1]
        tile_η, tile_ϕ = get_tile(tiling_setup, eta(flatjets, ijet), phi(flatjets, ijet))
        set_tile_index!(flatjets, ijet, get_tile_linear_index(tiling_setup, tile_η, tile_ϕ))
        push!(tiles[tile_η, tile_ϕ].jets, ijet)
    end
end


"""
Check that the state of tiling is consistent
- Every live pseudojet should be in one and only one tile
- Each tile should have the correct start point for its pseudojets
"""
function debug_tiles(tiles::Array{Tile, 2}, flatjets::FlatJets)
    msg = "Testing tile structures\n"
    jet_tile_index = similar(flatjets.kt2, Int)
    fill!(jet_tile_index, 0)
    for itile in eachindex(tiles)
        tile = tiles[itile]
        for ijet in tile.jets
            if jet_tile_index[ijet] != 0
                msg *= "Error: Found jet $jet already in tile $ijet\n"
            else
                jet_tile_index[ijet] = itile
            end
        end
    end
    for ijet in eachindex(jet_tile_index)
        if jet_tile_index[ijet] == 0
            msg *= "Error: Jet $ijet is not associated with a tile\n"
        end
    end
    msg
end

"""
Complete scan over all tiles to setup the nearest neighbour mappings at the beginning
"""
function find_all_tiled_nearest_neighbours!(tiles::Array{Tile, 2}, flatjets::FlatJets, tiling_setup::TilingDef, R2)
    # Iterate tile by tile...
    # tile_jet_list = Vector{Int}()
    for itile ∈ eachindex(tiles)
        itile_cartesian = get_tile_cartesian_indices(tiling_setup, itile)
        ## Debug for checking that my index calculations are correct
        # @assert itile_cartesian[1] == tiling_setup._tile_cartesian_indexes[itile][1] "$itile_cartesian -- $(tiling_setup._tile_cartesian_indexes[itile])"
        # @assert itile_cartesian[2] == tiling_setup._tile_cartesian_indexes[itile][2] "$itile_cartesian -- $(tiling_setup._tile_cartesian_indexes[itile])"

        # Take a Vector here, because we only iterate over the upper triangle of combinations
        # So it should be worth the cost of having an ordered collection
        # tile_jet_list = collect(tiles[itile].jets)
        # for (ijet_tile, ijet) in enumerate(tile_jet_list)
        #     for jjet in tile_jet_list[(ijet_tile+1):lastindex(tile_jet_list)]
        for ijet in tiles[itile].jets
            for jjet in tiles[itile].jets
                if ijet == jjet
                    continue
                end
                nn_dist = geometric_distance(eta(flatjets, ijet), phi(flatjets, ijet),
                    eta(flatjets, jjet), phi(flatjets, jjet))
                if nn_dist < nn_distance(flatjets, ijet)
                    set_nn_distance!(flatjets, ijet, nn_dist)
                    set_nearest_neighbour!(flatjets, ijet, jjet)
                end
                if nn_dist < nn_distance(flatjets, jjet)
                    set_nn_distance!(flatjets, jjet, nn_dist)
                    set_nearest_neighbour!(flatjets, jjet, ijet)
                end
            end

            # Now scan over rightmost neighbour tiles
            # This means moving in (η,ϕ) indexes, so now we need to map back from the 1D index
            # to the 2D one...
            for jtile_cartesian in rightmost_tiles(tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi, itile_cartesian[1], itile_cartesian[2])
                jtile = get_tile_linear_index(tiling_setup, jtile_cartesian[1], jtile_cartesian[2])
                ## Debug for checking that my index calculations are correct
                # @assert jtile == tiling_setup._tile_linear_indexes[jtile_cartesian[1], jtile_cartesian[2]]
                for jjet in tiles[jtile].jets
                    nn_dist = geometric_distance(eta(flatjets, ijet), phi(flatjets, ijet),
                        eta(flatjets, jjet), phi(flatjets, jjet))
                    if nn_dist < nn_distance(flatjets, ijet)
                        set_nn_distance!(flatjets, ijet, nn_dist)
                        set_nearest_neighbour!(flatjets, ijet, jjet)
                    end
                    if nn_dist < nn_distance(flatjets, jjet)
                        set_nn_distance!(flatjets, jjet, nn_dist)
                        set_nearest_neighbour!(flatjets, jjet, ijet)
                    end
                end
            end
        end
    end
    # Done with nearest neighbours, now calculate dij_distances
    for ijet in 1:length(flatjets.kt2)
        set_dij_distance!(flatjets, ijet, 
            get_dij_dist(
                nn_distance(flatjets, ijet), 
                kt2(flatjets, ijet), 
                nearest_neighbour(flatjets, ijet) == 0 ? 0.0 : kt2(flatjets, nearest_neighbour(flatjets, ijet)), 
                R2
            )
        )
                
    end
end

"""
Find neighbour for a particular jet index
"""
const updated_pair_jets = Int[] # If any reverese mappings change their NN distance, need to record this to update dij
function find_jet_nearest_neighbour!(tiles::Array{Tile, 2}, flatjets::FlatJets, tiling_setup::TilingDef, ijet::Int, R2)
    @debug "Finding neighbours of $ijet"
    set_nearest_neighbour!(flatjets, ijet, 0)
    set_nn_distance!(flatjets, ijet, R2)
    empty!(updated_pair_jets)
    # Jet's in own tile
    itile_cartesian = get_tile_cartesian_indices(tiling_setup, tile_index(flatjets, ijet))
    for jjet in tiles[tile_index(flatjets, ijet)].jets
        if ijet == jjet
            continue
        end
        nn_dist = geometric_distance(eta(flatjets, ijet), phi(flatjets, ijet),
            eta(flatjets, jjet), phi(flatjets, jjet))
        if nn_dist < nn_distance(flatjets, ijet)
            set_nn_distance!(flatjets, ijet, nn_dist)
            set_nearest_neighbour!(flatjets, ijet, jjet)
        end
        if nn_dist < nn_distance(flatjets, jjet)
            push!(updated_pair_jets, jjet)
            set_nn_distance!(flatjets, jjet, nn_dist)
            set_nearest_neighbour!(flatjets, jjet, ijet)
        end
    end
    for jtile_cartesian in neighbour_tiles(tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi, itile_cartesian[1], itile_cartesian[2])
        jtile = get_tile_linear_index(tiling_setup, jtile_cartesian[1], jtile_cartesian[2])
        for jjet in tiles[jtile].jets
            nn_dist = geometric_distance(eta(flatjets, ijet), phi(flatjets, ijet),
                eta(flatjets, jjet), phi(flatjets, jjet))
            if nn_dist < nn_distance(flatjets, ijet)
                set_nn_distance!(flatjets, ijet, nn_dist)
                set_nearest_neighbour!(flatjets, ijet, jjet)
            end
            if nn_dist < nn_distance(flatjets, jjet)
                push!(updated_pair_jets, jjet)
                set_nn_distance!(flatjets, jjet, nn_dist)
                set_nearest_neighbour!(flatjets, jjet, ijet)
            end
        end
    end
    set_dij_distance!(flatjets, ijet, 
        get_dij_dist(
            nn_distance(flatjets, ijet), 
            kt2(flatjets, ijet), 
            nearest_neighbour(flatjets, ijet) == 0 ? 0.0 : kt2(flatjets, nearest_neighbour(flatjets, ijet)), 
            R2
        )
    )
    for jjet in updated_pair_jets
        set_dij_distance!(flatjets, jjet, 
            get_dij_dist(
                nn_distance(flatjets, jjet), 
                kt2(flatjets, jjet), 
                kt2(flatjets, nearest_neighbour(flatjets, jjet)), 
                R2
            )
        )
    end
end

"""
Find all of the jets with a particular nearest neighbour
"""
const neighbours = Vector{Int}()
find_neighbours_of(flatjets::FlatJets, innv::Vector{Int}) = begin
    empty!(neighbours)
    @inbounds for (ijet, nn) in enumerate(flatjets.nearest_neighbour)
        if (nn in innv) && !(ijet in innv)
            push!(neighbours, ijet)
        end
    end
    # # Jets in the same tile
    # for ijet in tiles[tile_index(flatjets, inn)].jets
    #     if nearest_neighbour(flatjets, ijet) == inn
    #         push!(neighbours, ijet)
    #     end
    # end
    # # Jets in neighbouring tiles
    # itile_cartesian = get_tile_cartesian_indices(tiling_setup, inn)
    # for jtile_cartesian in neighbour_tiles(tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi, itile_cartesian[1], itile_cartesian[2])
    #     for jjet in tiles[jtile_cartesian[1], jtile_cartesian[2]].jets
    #         if nearest_neighbour(flatjets, jjet) == inn
    #             push!(neighbours, jjet)
    #         end
    #     end
    # end
    neighbours
end

find_neighbours_of(flatjets::FlatJets, inn::Int) = begin
    empty!(neighbours)
    @inbounds for (ijet, nn) in enumerate(flatjets.nearest_neighbour)
        if nn == inn
            push!(neighbours, ijet)
        end
    end
    neighbours
end

"""
Look for any jets which had a shuffled jet as their nearest neighbour, which has
moved from old_index to new_index
"""
function move_nn_index!(flatjets::FlatJets, old_index, new_index)
    # Do this in a simple loop - it's contiguous memory, very fast and can be vectorised
    @debug "Moving jet $old_index to $new_index"
    @turbo for ijet ∈ eachindex(flatjets.nearest_neighbour)
        moveme = flatjets.nearest_neighbour[ijet] == old_index
        flatjets.nearest_neighbour[ijet] = moveme ? new_index : flatjets.nearest_neighbour[ijet]
    end
end

"""
Find the index in the vector which has the lowest value of dij
"""
function find_lowest_dij_index(dij_distances)
    imin = 0
    vmin = typemax(typeof(dij_distances[1]))
    @turbo for i ∈ eachindex(dij_distances)
        newmin = dij_distances[i] < vmin
        vmin = newmin ? dij_distances[i] : vmin
        imin = newmin ? i : imin
    end
    imin
end

"""
Tiled jet reconstruction
"""
function tiled_jet_reconstruct(objects::AbstractArray{T}; p = -1, R = 1.0, recombine = +) where T
	# bounds
	N::Int = length(objects)
	@debug "Initial particles: $(N)"

	# params
	R2::Float64 = R * R
	p = (round(p) == p) ? Int(p) : p # integer p if possible
	ap = abs(p) # absolute p

	# Input data
	jet_objects = copy(objects) # We will append merged jets, so we need a copy
	_kt2 = (JetReconstruction.pt.(objects) .^ 2) .^ p
	_phi = JetReconstruction.phi.(objects)
	_eta = JetReconstruction.eta.(objects)
	_jet_index = collect(1:N) # Initial jets are just numbered 1:N, mapping directly to jet_objects
    # println(_jet_index)

	# Each jet stores which tile it is in, so need the usual container for that
	_tile_index = zeros(Int, N)
	# sizehint!(tile_index, N * 2)

	# Linked list: this is the index of the next/previous jet for this tile (0 = the end/beginning)
	# _next_jet = zeros(Int, N)
    # _prev_jet = zeros(Int, N)

    # Nearest neighbour parameters
    _nearest_neighbour = zeros(Int, N)
    _nn_distance = fill(1.0e9, N)
    _dij_distance = fill(1.0e9, N)

	# returned values
	jets = T[] # result
	sequences = Vector{Int}[[x] for x in 1:N]

	flatjets = FlatJets(_kt2, _eta, _phi, _jet_index, _tile_index, 
                        # _next_jet, _prev_jet,
                        _nearest_neighbour, _nn_distance, _dij_distance)
    # println(flatjets.jet_index)

	# Tiling
	tiling_setup= setup_tiling(_eta, R)
	@debug("Tiles: $(tiling_setup._n_tiles_eta)x$(tiling_setup._n_tiles_phi)")

	# Setup the tiling array
	tiles = Array{Tile,2}(undef, (tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi))
	@inbounds for itile in eachindex(tiles)
		tiles[itile] = Tile()
	end

    # Populate tile jet lists, from the initial particles
	populate_tile_lists!(tiles, flatjets, tiling_setup)

    # Useful state debugging
    # print(debug_tiles(tiles, flatjets))

	# Setup initial nn, nndist and dij values
	find_all_tiled_nearest_neighbours!(tiles, flatjets, tiling_setup, R2)

    # A few allocations outside the loop
    # itouched_tiles = Set{Int}()
    # tainted_indexes = Set{Int}()
    itouched_jets = Set{Int}()

	# At each iteration we either merge two jets to one, or finalise a jet
	# Thus each time we lose one jet, and it therefore takes N iterations to complete
	# the algorithm
	for iteration in 1:N
        @debug "Iteration $(iteration) - Active Jets $(lastindex(flatjets.kt2))"

        iclosejetA = find_lowest_dij_index(flatjets.dij_distance)
        iclosejetB = nearest_neighbour(flatjets, iclosejetA)
        @debug "Closest jets $iclosejetA, $iclosejetB: $(dij_distance(flatjets, iclosejetA))"

        # Finalise jet or merge jets?
        if iclosejetB != 0
            # Merge jets A and B - jet-jet recombination
            # If necessary relabel A & B to ensure jetB < jetA, that way if
            # the larger of them == newtail then that ends up being jetA and
            # the new jet that is added as jetB is inserted in a position that
            # has a future!
            if iclosejetA < iclosejetB
                iclosejetA, iclosejetB = iclosejetB, iclosejetA
            end
            @debug "Merge jets: jet indexes $(jet_index(flatjets, iclosejetA)) in $(get_tile_cartesian_indices(tiling_setup, tile_index(flatjets, iclosejetA))), $(jet_index(flatjets, iclosejetB)) in $(get_tile_cartesian_indices(tiling_setup, tile_index(flatjets, iclosejetB)))"
            newjet = recombine(jet_objects[jet_index(flatjets, iclosejetA)], 
                               jet_objects[jet_index(flatjets, iclosejetB)])
            push!(jet_objects, newjet)
            inewjet = lastindex(jet_objects)
            ## Set the _sequence for the two merged jets, which is the merged jet index
			push!(sequences, [iclosejetA, iclosejetB, inewjet])
			push!(sequences[jet_index(flatjets, iclosejetA)], inewjet)
			push!(sequences[jet_index(flatjets, iclosejetB)], inewjet)
            # Remove merged jets from their tile jet lists
            delete!(tiles[tile_index(flatjets, iclosejetA)].jets, iclosejetA)
            delete!(tiles[tile_index(flatjets, iclosejetB)].jets, iclosejetB)
            # Now find out which jets had A or B as their nearest neighbour - they will 
            # need to be rescanned
            empty!(itouched_jets)
            union!(itouched_jets, find_neighbours_of(flatjets, [iclosejetA, iclosejetB]))
            @debug "Jets to update from A/B neighbours: $(itouched_jets)"
            ####
            # Now push the newjet into jetB's slot, add it to its tile's jet list and to
            # the set of jets that need to rescan their neighbours
            @debug "Adding merged jet to slot $iclosejetB"
            insert_flatjet!(flatjets, tiling_setup, p, iclosejetB, inewjet, newjet)
            push!(tiles[tile_index(flatjets, iclosejetB)].jets, iclosejetB)
            push!(itouched_jets, iclosejetB) # This is the newjet!
            # Now kill jetA, shuffling if needed
            shuffled_jet = suppress_flatjet!(flatjets, iclosejetA)
            @debug "Killed jet $iclosejetA and shuffled jet $shuffled_jet"
            # If we had a shuffled jet, we need to update any jet who had this jet as their
            # neighbour
            if shuffled_jet != 0
                delete!(tiles[tile_index(flatjets, iclosejetA)].jets, shuffled_jet)
                push!(tiles[tile_index(flatjets, iclosejetA)].jets, iclosejetA)
                move_nn_index!(flatjets, shuffled_jet, iclosejetA)
                if shuffled_jet in itouched_jets
                    delete!(itouched_jets, shuffled_jet)
                    push!(itouched_jets, iclosejetA)
                end
            end
            ## Updates
            for ijet in itouched_jets
                find_jet_nearest_neighbour!(tiles, flatjets, tiling_setup, ijet, R2)
            end
        else
            # Finalise jet A
            @debug "Finalise jet: jet index $(jet_index(flatjets, iclosejetA))"
            push!(sequences[jet_index(flatjets, iclosejetA)], 0)
            push!(jets, jet_objects[jet_index(flatjets, iclosejetA)])
            # Remove finalsied jet from its tile jet lists
            delete!(tiles[tile_index(flatjets, iclosejetA)].jets, iclosejetA)
            # Now find out which jets had A as their nearest neighbour - they will 
            # need to be rescanned
            empty!(itouched_jets)
            union!(itouched_jets, find_neighbours_of(flatjets, iclosejetA))
            @debug "Jets to update from A neighbours: $(itouched_jets)"
            # Now kill jetA, shuffling if needed
            shuffled_jet = suppress_flatjet!(flatjets, iclosejetA)
            @debug "Killed jet $iclosejetA and shuffled jet $shuffled_jet"
            # If we had a shuffled jet, we need to update any jet who had this jet as their
            # neighbour
            if shuffled_jet != 0
                delete!(tiles[tile_index(flatjets, iclosejetA)].jets, shuffled_jet)
                push!(tiles[tile_index(flatjets, iclosejetA)].jets, iclosejetA)
                move_nn_index!(flatjets, shuffled_jet, iclosejetA)
                if shuffled_jet in itouched_jets
                    delete!(itouched_jets, shuffled_jet)
                    push!(itouched_jets, iclosejetA)
                end
            end
            ## Updates
            for ijet in itouched_jets
                find_jet_nearest_neighbour!(tiles, flatjets, tiling_setup, ijet, R2)
            end
        end
    end

    # The sequences return value is a list of all jets that merged to this one
	jets, sequences
end
