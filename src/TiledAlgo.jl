# Tiled jet reconstruction, linked list data structure approach

using Logging

"""
Tiled jet reconstruction
"""
function tiled_jet_reconstruct(objects::AbstractArray{T}; p = -1, R = 1.0, recombine = +) where T
	# bounds
	N::Int = length(objects)
	@debug "Initial particles: $(N)"

	# params
	_R2::Float64 = R * R
	_p = (round(p) == p) ? Int(p) : p # integer p if possible
	ap = abs(_p) # absolute p

	# Input data
	_objects = copy(objects)
	sizehint!(_objects, N * 2)
	_kt2 = (JetReconstruction.pt.(_objects) .^ 2) .^ _p
	sizehint!(_kt2, N * 2)
	_phi = JetReconstruction.phi.(_objects)
	sizehint!(_phi, N * 2)
	_eta = JetReconstruction.eta.(_objects)
	sizehint!(_eta, N * 2)
	_index = collect(1:N) # Initial jets are just numbered 1:N
	sizehint!(_index, N * 2)

	# returned values
	jets = T[] # result
	_sequences = Vector{Int}[[x] for x in 1:N]

	flat_jets = FlatJetSoA(N, _kt2, _eta, _phi, _index)

	# Tiling
	tiling_setup, tile_jets = setup_tiling(_eta, R)

	# # Populate tiles, from the initial particles
	# populate_tiles!(tile_jets, tiling_setup, flat_jets, _R2)

	# # Setup initial nn, nndist and dij values
	# min_dij_itile, min_dij_ijet, min_dij = find_all_nearest_neighbours!(tile_jets, tiling_setup, flat_jets, _R2)

	# # Move some variables outside the loop, to avoid per-loop allocations
	# itouched_tiles = Set{Int}()
	# sizehint!(itouched_tiles, 12)
	# tainted_slots = Set{TiledNN}()
	# sizehint!(tainted_slots, 4)

	# # At each iteration we either merge two jets to one, or finalise a jet
	# # Thus each time we lose one jet, and it therefore takes N iterations to complete
	# # the algorithm
	# for iteration in 1:N
    #     @debug "Iteration $(iteration)"

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
	# 		index_tile_jetA = TiledNN(min_dij_itile, min_dij_ijet)
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
	# 		index_tile_jetA = TiledNN(min_dij_itile, min_dij_ijet)
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
	# 		push!(flat_jets._index, length(_objects))
	# 		push!(flat_jets._phi, JetReconstruction.phi(merged_jet))
	# 		push!(flat_jets._eta, JetReconstruction.eta(merged_jet))
	# 		push!(flat_jets._kt2, (JetReconstruction.pt(merged_jet)^2)^_p)
	# 		merged_jet_index = lastindex(_objects)

	# 		ieta_merged_jet, iphi_merged_jet = get_tile(tiling_setup, flat_jets._eta[merged_jet_index],
	# 			flat_jets._phi[merged_jet_index])
	# 		itile_merged_jet = tiling_setup._tile_linear_indexes[ieta_merged_jet, iphi_merged_jet]

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
	# 			insert_jet!(tile_jets[itile_merged_jet], index_tile_jetA._ijet, merged_jet_index, flat_jets, _R2)
	# 			index_tile_merged_jet = TiledNN(itile_merged_jet, index_tile_jetA._ijet)
	# 			# Now zap jetB
	# 			push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetB))
	# 		elseif itile_merged_jet == index_tile_jetB._itile
	# 			# Use jetB's slot
	# 			insert_jet!(tile_jets[itile_merged_jet], index_tile_jetB._ijet, merged_jet_index, flat_jets, _R2)
	# 			index_tile_merged_jet = TiledNN(itile_merged_jet, index_tile_jetB._ijet)
	# 			# Now zap jetA
	# 			push!(tainted_slots, remove_jet!(tile_jets, index_tile_jetA))
	# 		else
    #             # Merged jet is in a different tile
    #             add_jet!(tile_jets[itile_merged_jet], merged_jet_index, flat_jets, _R2)
    #             index_tile_merged_jet = TiledNN(itile_merged_jet, tile_jets[itile_merged_jet]._size)
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
	# 				tile._nn[ijet] = TiledNN(0, 0)
	# 				tile._nndist[ijet] = _R2
	# 				scan_neighbors!(tile_jets, TiledNN(itouched_tile, ijet), _R2)
	# 			end
	# 		end
	# 	end
	# end
    # The sequences return value is a list of all jets that merged to this one
	jets, _sequences
end
