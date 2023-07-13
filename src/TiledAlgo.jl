# Tiled jet reconstruction, linked list data structure approach

using Logging

"""
Structure holding the flat jets for a tiled reconstruction
"""
mutable struct FlatJets
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
function suppress_jet!(jets::FlatJets, n::Int)
    # Is the jet we want to get rid of the final jet? In this case the job is trivial
    ilast = size(jets.kt2)[1]
    tainted_index::Int = 0
    if n != ilast
        # Not the last jet - need to shuffle...
        set_kt2!(jets, n, kt2(jets, ilast))
        set_eta!(jets, n, eta(jets, ilast))
        set_phi!(jets, n, phi(jets, ilast))
        set_jet_index!(jets, n, jet_index(jets, ilast))
        set_tile_index!(jets, n, tile_index(jets, ilast))
        ### TO BE CHECKED
        # set_next_jet!(jets, n, 0)
        # set_prev_jet!(jets, n, 0)
        set_nearest_neighbour!(jets, n, nearest_neighbour(jets, ilast))
        set_nn_distance!(jets, n, nn_distance(jets, ilast))
        set_dij_distance!(jets, n, dij_distance(jets, ilast))
        tainted_index = ilast
    else

    end
    pop!(jets.kt2)
    pop!(jets.eta)
    pop!(jets.phi)
    pop!(jets.jet_index)
    pop!(jets.tile_index)
    pop!(jets.nearest_neighbour)
    pop!(jets.nn_distance)
    pop!(jets.dij_distance)
    tainted_index
end


"""
Structure holding the tiles for the reconstruction
"""
 mutable struct Tile
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
        itile_cartesian = get_tile_cartesian_indices(tiling_setup, itile)
        ## Debug for checking that my index calculations are correct
        # @assert itile_cartesian[1] == tiling_setup._tile_cartesian_indexes[itile][1] "$itile_cartesian -- $(tiling_setup._tile_cartesian_indexes[itile])"
        # @assert itile_cartesian[2] == tiling_setup._tile_cartesian_indexes[itile][2] "$itile_cartesian -- $(tiling_setup._tile_cartesian_indexes[itile])"

        # Take a Vector here, because we only iterate over the upper triangle of combinations
        # So it should be worth the cost of having an ordered collection
        tile_jet_list = Vector{Int}(collect(tiles[itile].jets))
        for (ijet_tile, ijet) in enumerate(tile_jet_list)
            for jjet in tile_jet_list[(ijet_tile+1):lastindex(tile_jet_list)]
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

    # Populate tile linked lists, from the initial particles
	populate_tile_lists!(tiles, flatjets, tiling_setup)

    # Useful state debugging
    # print(debug_tiles(tiles, flatjets))

	# Setup initial nn, nndist and dij values
	find_all_tiled_nearest_neighbours!(tiles, flatjets, tiling_setup, R2)

    # A few allocations outside the loop
    itouched_tiles = Set{Int}()
    tainted_indexes = Set{Int}()

	# At each iteration we either merge two jets to one, or finalise a jet
	# Thus each time we lose one jet, and it therefore takes N iterations to complete
	# the algorithm
	# for iteration in 1:N
    #     @debug "Iteration $(iteration)"

    #     # Find the lowest value of dij_distance
    #     iclosejetA = argmin(flatjets.dij_distance)
    #     iclosejetB = nearest_neighbour(flatjets, iclosejetA)
    #     @debug "Closest jets $iclosejetA, $iclosejetB: $(kt2(flatjets, iclosejetA))"

    #     # Finalise jet or merge jets?
    #     if iclosejetB != 0
    #         # Merge jets A and B - jet-jet recombination
    #         # If necessary relabel A & B to ensure jetB < jetA, that way if
    #         # the larger of them == newtail then that ends up being jetA and
    #         # the new jet that is added as jetB is inserted in a position that
    #         # has a future!
    #         if iclosejetA < iclosejetB
    #             iclosejetA, iclosejetB = iclosejetB, iclosejetA
    #         end
    #         @debug "Jet indexes $(jet_index(flatjets, iclosejetA)), $(jet_index(flatjets, iclosejetB))"
    #         newjet = recombine(jet_objects[jet_index(flatjets, iclosejetA)], 
    #                            jet_objects[jet_index(flatjets, iclosejetB)])
    #         println(newjet)
    #         println(jet_objects[iclosejetA])
    #         println(jet_objects[iclosejetB])
    #         push!(jet_objects, newjet)
    #         inewjet = size(jet_objects)[1]
    #         # Keep track of touched tiles and tainted neighbour indexes - these have to
    #         # be reevaluated
    #         push!(itouched_tiles, tile_index(flatjets, iclosejetA), tile_index(flatjets, iclosejetB))
    #         push!(tainted_indexes, iclosejetA, iclosejetB)
    #         @debug "$(itouched_tiles) $(tainted_indexes)"
    #         # Now push the newjet into jetB's slot
    #         insert_flatjet!(flatjets, tiling_setup, p, iclosejetB, inewjet, newjet)
    #         # Now kill jetA, shuffling if needed
    #         tainted_index = suppress_flatjet!(flatjets, iclosejetA)
    #         if tainted_index != 0
    #             push!(tainted_indexes, iclosejetA)
    #         end
    #     else
    #         # Finalise jet A
    #     end

    #     # exit(0)
    # end

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
