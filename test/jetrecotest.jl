#! /usr/bin/env julia

using JetReconstruction
using Test
using JSON
using IsApprox
using IsApprox: iszero

"""Read JSON file with fastjet jets in it"""
function read_fastjet_outputs(;fname="test/data/jet_collections_fastjet.json")
    f = open(fname)
    JSON.parse(f)
end

"""Sort jet outputs by pt of final jets"""
function sort_jets!(event_jet_array)
    jet_pt(jet) = jet["pt"]
    for e in event_jet_array
        sort!(e["jets"], by=jet_pt, rev=true)
    end
end

function sort_jets!(jet_array::Vector{FinalJet})
    jet_pt(jet) = jet.pt
    sort!(jet_array, by=jet_pt, rev=true)
end

function main()
    # Read our fastjet outputs
    fastjet_jets = read_fastjet_outputs()
    sort_jets!(fastjet_jets)

    # Test each stratgy...
    do_jet_test(N2Plain, fastjet_jets)
    do_jet_test(N2Tiled, fastjet_jets)

end

function do_jet_test(strategy::JetRecoStrategy, fastjet_jets;
    ptmin::Float64 = 5.0,
	distance::Float64 = 0.4,
	power::Integer = -1)

	# Strategy
	if (strategy == N2Plain)
		jet_reconstruction = sequential_jet_reconstruct
        strategy_name = "N2Plain"
	elseif (strategy == N2Tiled)
		jet_reconstruction = tiled_jet_reconstruct
        strategy_name = "N2Tiled"
	else
		throw(ErrorException("Strategy not yet implemented"))
	end

    # Now run our jet reconstruction...
    events::Vector{Vector{PseudoJet}} = read_final_state_particles("test/data/events.hepmc3")
    event_vector = pseudojets2vectors(events)
    jet_collection = FinalJets[]
    for (ievt, event) in enumerate(event_vector)
        finaljets, _ = jet_reconstruction(event, R=distance, p=power)
        fj = final_jets(finaljets, ptmin)
        sort_jets!(fj)
        push!(jet_collection, FinalJets(ievt, fj))
    end

    @testset "Jet Reconstruction, $strategy_name" begin
        # Test each event in turn...
        for (ievt, event) in enumerate(jet_collection)
            @testset "Event $(ievt)" begin
                @test size(event.jets) == size(fastjet_jets[ievt]["jets"])
                # Test each jet in turn
                for (ijet, jet) in enumerate(event.jets)
                    if ijet <= size(fastjet_jets[ievt]["jets"])[1]
                        # Approximate test - note that ≈ itself is just too strict here
                        # There's no a≈b here, so we test the difference is ≈0
                        @test iszero(jet.rap - fastjet_jets[ievt]["jets"][ijet]["rap"], Approx(atol=1e-7))
                        @test iszero(jet.phi - fastjet_jets[ievt]["jets"][ijet]["phi"], Approx(atol=1e-7))
                        @test iszero(jet.pt - fastjet_jets[ievt]["jets"][ijet]["pt"], Approx(atol=1e-7))
                    end
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end
