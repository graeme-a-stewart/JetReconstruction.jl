# Tiled jet reconstruction, linked list data structure approach

using Logging

"""
Structure holding the flat jets for a tiled reconstruction
"""
mutable struct FlatJets
    # Physics quantities
	kt2::Vector{Float64}       	# p_t^-2*power
	eta::Vector{Float64}        # Rapidity
	phi::Vector{Float64}        # Phi coordinate

    # Tiles and linked list 
	# tile_index::Vector{Int}    # My tile index
	next_jet::Vector{Int}      # This is the linked list index of the next jet for this tile (0→end)
    prev_jet::Vector{Int}      # This is the linked list index of the previous jet for this tile (0→first)

    # Reconstruction parameters
    nearest_neighbour::Vector{Int}
    nn_distance::Vector{Float64}
    dij_distance::Vector{Float64}
end

# Accessor functions for the jet array
kt2(jets::FlatJets, n::Int) = jets.kt2[n]
eta(jets::FlatJets, n::Int) = jets.eta[n]
phi(jets::FlatJets, n::Int) = jets.phi[n]
# tile_index(jets::FlatJets, n::Int) = jets.tile_index[n]
next_jet(jets::FlatJets, n::Int) = jets.next_jet[n]
prev_jet(jets::FlatJets, n::Int) = jets.prev_jet[n]
nearest_neighbour(jets::FlatJets, n::Int) = jets.nearest_neighbour[n]
nn_distance(jets::FlatJets, n::Int) = jets.nn_distance[n]
dij_distance(jets::FlatJets, n::Int) = jets.dij_distance[n]

# Setters
# set_tile_index!(jets::FlatJets, n::Int, i) = jets.tile_index[n] = i
set_next_jet!(jets::FlatJets, n::Int, next::Int) = jets.next_jet[n] = next
set_prev_jet!(jets::FlatJets, n::Int, prev::Int) = jets.prev_jet[n] = prev 

set_nearest_neighbour!(jets::FlatJets, n::Int, v::Int) = jets.nearest_neighbour[n] = v
set_nn_distance!(jets::FlatJets, n::Int, v::Float64) = jets.nn_distance[n] = v
set_dij_distance!(jets::FlatJets, n::Int, v::Float64) = jets.dij_distance[n] = v


"""
Structure holding the tiles for the reconstruction
"""
 mutable struct Tile
    first_jet::Int
end
first_jet(t::Tile) = t.first_jet
set_first_jet!(t::Tile, first::Int) = t.first_jet = first

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
        # This is quite expensive, like ~12us, do we really need it?
        # set_tile_index!(flatjets, ijet, tiling_setup._tile_linear_indexes[tile_η, tile_ϕ])
        if tiles[tile_η, tile_ϕ].first_jet == 0
            # First jet in this tile
            tiles[tile_η, tile_ϕ].first_jet = ijet
        else
            # Insert this jet at the beginning of the list for this tile
            old_first_jet = first_jet(tiles[tile_η, tile_ϕ])
            set_first_jet!(tiles[tile_η, tile_ϕ], ijet)
            set_next_jet!(flatjets, ijet, old_first_jet)
            set_prev_jet!(flatjets, old_first_jet, ijet)
        end
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
        ijet = first_jet(tile)
        while ijet != 0
            if jet_tile_index[ijet] != 0
                msg *= "Error: Found jet $jet already in tile $ijet\n"
            else
                jet_tile_index[ijet] = itile
            end
            ijet = next_jet(flatjets, ijet)
        end
    end
    for ijet in eachindex(jet_tile_index)
        if jet_tile_index[ijet] == 0
            msg *= "Error: Jet $ijet is not associated with a tile"
        end
    end
    msg
end

"""
Complete scan over all tiles to setup the nearest neighbour mappings at the beginning
"""
function find_all_tiled_nearest_neighbours!(tiles::Array{Tile, 2}, flatjets::FlatJets, tiling_setup::TilingDef, R2)
    # Iterate tile by tile...
    for itile in eachindex(tiles)
        itile_cartesian = tiling_setup._tile_cartesian_indexes[itile]
        ijet = first_jet(tiles[itile])
        # println("$itile $itile_cartesian - $ijet")
        if ijet == 0
            continue
        end
        jjet = next_jet(flatjets, ijet)
        # print(" $jjet")
        # Scan over all the other jets in my tile
        while jjet != 0
            # As this is symetric, we do not need to care about jets earlier in the list
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
            # Move to next jet in the list to compare against ("RHS")
            jjet = next_jet(flatjets, jjet)
        end
        # println()

        # Now scan over rightmost neighbour tiles
        # This means moving in (η,ϕ) indexes, so now we need to map back from the 1D index
        # to the 2D one...
        for jtile_cartesian in rightmost_tiles(tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi, itile_cartesian[1], itile_cartesian[2])
            jtile = tiling_setup._tile_linear_indexes[jtile_cartesian[1], jtile_cartesian[2]]
            jjet = first_jet(tiles[jtile])
            while jjet != 0
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
                # Move to next jet in the list to compare against ("RHS")
                jjet = next_jet(flatjets, jjet)
            end
        end
    end
    # Done with nearest neighbours, now calculate dij_distances
    for ijet in 1:size(flatjets.kt2)[1]
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
	# jet_objects = copy(objects) # Don't need to copy this, we don't touch it
	_kt2 = (JetReconstruction.pt.(objects) .^ 2) .^ p
	_phi = JetReconstruction.phi.(objects)
	_eta = JetReconstruction.eta.(objects)
	# index = collect(1:N) # Initial jets are just numbered 1:N

	# Each jet stores which tile it is in, so need the usual container for that # Needed?
	# tile_index = zeros(Int, N)
	# sizehint!(tile_index, N * 2)

	# Linked list: this is the index of the next/previous jet for this tile (0 = the end/beginning)
	_next_jet = zeros(Int, N)
    _prev_jet = zeros(Int, N)

    # Nearest neighbour parameters
    _nearest_neighbour = zeros(Int, N)
    _nn_distance = fill(1.0e9, N)
    _dij_distance = fill(1.0e9, N)

	# returned values
	jets = T[] # result
	sequences = Vector{Int}[[x] for x in 1:N]

	flatjets = FlatJets(_kt2, _eta, _phi, _next_jet, _prev_jet,
                        _nearest_neighbour, _nn_distance, _dij_distance)

	# Tiling
	tiling_setup= setup_tiling(_eta, R)
	@debug("Tiles: $(tiling_setup._n_tiles_eta)x$(tiling_setup._n_tiles_phi)")

	# Setup the tiling array
	tiles = Array{Tile,2}(undef, (tiling_setup._n_tiles_eta, tiling_setup._n_tiles_phi))
	@inbounds for itile in eachindex(tiles)
		tiles[itile] = Tile(0)
	end

    # Populate tile linked lists, from the initial particles
	populate_tile_lists!(tiles, flatjets, tiling_setup)

    # Useful state debugging
    # print(debug_tiles(tiles, flatjets))

	# Setup initial nn, nndist and dij values
	find_all_tiled_nearest_neighbours!(tiles, flatjets, tiling_setup, R2)

	# At each iteration we either merge two jets to one, or finalise a jet
	# Thus each time we lose one jet, and it therefore takes N iterations to complete
	# the algorithm
	for iteration in 1:N
        @debug "Iteration $(iteration)"

        # Find the lowest value of dij_distance
        iclosejetA = argmin(flatjets.dij_distance)
        iclosejetB = nearest_neighbour(flatjets, iclosejetA)
        @debug "Closest jets $iclosejetA, $iclosejetB: $(kt2(flatjets, iclosejetA))"
        break

        # Finalise jet or merge jets?
        if iclosejetB != 0
            # Merge jets A and B
        else
            # Finalise jet A
        end
    end

	# 	# For the first iteration the nearest neighbour is known
	# 	if iteration > 1
	# 		# Now find again the new nearest dij jets
	# 		min_dij = 1.0e20
	# 		min_dij_itile = 0
	# 		min_dij_ijet = 0
	# 		@inbounds for itile in eachindex(tile_jets)
	# 			@inbounds for ijet in 1:tile_jets[itile]._size
	# 				if tile_jets[itile]._dij[ijet] < min_dij
	# 					min_dij_itile = itile
	# 					min_dij_ijet = ijet
	# 					min_dij = tile_jets[itile]._dij[ijet]
	# 				end
	# 			end
	# 		end
	# 	end

    #     @debug "$(min_dij) at ($(min_dij_itile), $(min_dij_ijet)) $(tile_jets[min_dij_itile]._index[min_dij_ijet]) -> $(tile_jets[min_dij_itile]._nn[min_dij_ijet])"
	# 	# Is this a merger or a final jet?
	# 	if tile_jets[min_dij_itile]._nn[min_dij_ijet]._itile == 0
	# 		# Final jet
    #         jet_merger = false
	# 		index_tile_jetA = TiledSoACoord(min_dij_itile, min_dij_ijet)
	# 		index_jetA = tile_jets[min_dij_itile]._index[min_dij_ijet]
	# 		empty!(tainted_slots)
	# 		push!(tainted_slots, index_tile_jetA)
    #         push!(jets, _objects[index_jetA])
	# 		push!(_sequences[index_jetA], 0)
	# 		@debug "Finalise jet $(tile_jets[min_dij_itile]._index[min_dij_ijet]) ($(_sequences[index_jetA])) $(JetReconstruction.pt(_objects[index_jetA]))"
	# 		push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetA))
	# 	else
	# 		# Merge two jets
    #         jet_merger = true
	# 		index_tile_jetA = TiledSoACoord(min_dij_itile, min_dij_ijet)
	# 		index_tile_jetB = tile_jets[min_dij_itile]._nn[min_dij_ijet]
	# 		index_jetA = tile_jets[min_dij_itile]._index[min_dij_ijet]
	# 		index_jetB = nnindex(tile_jets, min_dij_itile, min_dij_ijet)
	# 		@debug "Merge jets $(index_jetA) ($(index_tile_jetA)) and $(index_jetB) ($(index_tile_jetB))"
	# 		merged_jet = recombine(_objects[index_jetA], _objects[index_jetB])

    #         # If A and B are in the same tile, ensure that A is the earlier slot
    #         # so that slots are filled up correctly
    #         if (index_tile_jetA._itile == index_tile_jetB._itile) && (index_tile_jetA._ijet > index_tile_jetB._ijet)
    #             index_tile_jetA, index_tile_jetB = index_tile_jetB, index_tile_jetA
    #             index_jetA, index_jetB = index_jetB, index_jetA
    #         end

	# 		push!(_objects, merged_jet)
	# 		push!(flatjets._index, length(_objects))
	# 		push!(flatjets._phi, JetReconstruction.phi(merged_jet))
	# 		push!(flatjets._eta, JetReconstruction.eta(merged_jet))
	# 		push!(flatjets._kt2, (JetReconstruction.pt(merged_jet)^2)^_p)
	# 		merged_jet_index = lastindex(_objects)

	# 		iη_merged_jet, iϕ_merged_jet = get_tile(tiling_setup, flatjets._eta[merged_jet_index],
	# 			flatjets._phi[merged_jet_index])
	# 		itile_merged_jet = tiling_setup._tile_linear_indexes[iη_merged_jet, iϕ_merged_jet]

	# 		# Set the _sequence for the two merged jets, which is the merged jet index
	# 		push!(_sequences, [_sequences[index_jetA]; _sequences[index_jetB]; merged_jet_index])
	# 		push!(_sequences[index_jetA], merged_jet_index)
	# 		push!(_sequences[index_jetB], merged_jet_index)


	# 		# Delete jetA and jetB from their tiles
	# 		empty!(tainted_slots)
	# 		push!(tainted_slots, index_tile_jetA)
	# 		push!(tainted_slots, index_tile_jetB)
	# 		if itile_merged_jet == index_tile_jetA._itile
	# 			# Put the new jet into jetA's slot
	# 			insert_jet!(tile_jets[itile_merged_jet], index_tile_jetA._ijet, merged_jet_index, flatjets, _R2)
	# 			index_tile_merged_jet = TiledSoACoord(itile_merged_jet, index_tile_jetA._ijet)
	# 			# Now zap jetB
	# 			push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetB))
	# 		elseif itile_merged_jet == index_tile_jetB._itile
	# 			# Use jetB's slot
	# 			insert_jet!(tile_jets[itile_merged_jet], index_tile_jetB._ijet, merged_jet_index, flatjets, _R2)
	# 			index_tile_merged_jet = TiledSoACoord(itile_merged_jet, index_tile_jetB._ijet)
	# 			# Now zap jetA
	# 			push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetA))
	# 		else
    #             # Merged jet is in a different tile
    #             add_jet!(tile_jets[itile_merged_jet], merged_jet_index, flatjets, _R2)
    #             index_tile_merged_jet = TiledSoACoord(itile_merged_jet, tile_jets[itile_merged_jet]._size)
    #             # Now zap both A and B
    #             push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetA))
    #             push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetB))
	# 		end

	# 		# For our new merged jet, scan for nearest neighbours
	# 		# Remember, this is pair-wise, so it will update all jets in its tile and neighbours
	# 		scan_neighbors!(tile_jets, index_tile_merged_jet, _R2)
	# 	end

    #     # Now take care of tainted neighbours
	# 	empty!(itouched_tiles)
    #     push!(itouched_tiles, index_tile_jetA._itile)
	# 	union!(itouched_tiles, tile_jets[index_tile_jetA._itile]._nntiles)
    #     if (jet_merger && index_tile_jetB._itile != index_tile_jetA._itile)
    #         push!(itouched_tiles, index_tile_jetB._itile)
    #         union!(itouched_tiles, tile_jets[index_tile_jetB._itile]._nntiles)
    #     end

	# 	# Scan over the touched tiles, look for jets whose _nn is tainted
	# 	@inbounds for itouched_tile in itouched_tiles
	# 		tile = tile_jets[itouched_tile]
	# 		@inbounds for ijet in 1:tile._size
	# 			if tile._nn[ijet] in tainted_slots
	# 				tile._nn[ijet] = TiledSoACoord(0, 0)
	# 				tile._nndist[ijet] = _R2
	# 				scan_neighbors!(tile_jets, TiledSoACoord(itouched_tile, ijet), _R2)
	# 			end
	# 		end
	# 	end
	# end
    # The sequences return value is a list of all jets that merged to this one
	jets, sequences
end
