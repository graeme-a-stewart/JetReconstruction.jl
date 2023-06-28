#! /usr/bin/env julia

using JetReconstruction
using Test
using JSON

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

    # Now run our jet reconstruction...
    jet_reconstruction = sequential_jet_reconstruct
    events::Vector{Vector{PseudoJet}} = read_final_state_particles("test/data/events.hepmc3")
    event_vector = pseudojets2vectors(events)
    jet_collection = FinalJets[]
    for (ievt, event) in enumerate(event_vector)
        finaljets, _ = jet_reconstruction(event, R=0.4, p=-1)
        fj = final_jets(finaljets, 5.0)
        sort_jets!(fj)
        # println(fj)
        push!(jet_collection, FinalJets(ievt, fj))
    end

    @testset verbose = true "Jet Reconstruction" begin
        # Test each event in turn...
        for (ievt, event) in enumerate(jet_collection)
            @testset "Event $(ievt)" begin
                @test size(event.jets) == size(fastjet_jets[ievt]["jets"])
                # Test each jet in turn
                for (ijet, jet) in enumerate(event.jets)
                    if ijet <= size(fastjet_jets[ievt]["jets"])[1]
                        # Approximat
                        # @test jet.rap ≈ fastjet_jets[ievt]["jets"][ijet]["rap"]
                        # @test jet.phi ≈ fastjet_jets[ievt]["jets"][ijet]["phi"]
                        @test jet.pt ≈ fastjet_jets[ievt]["jets"][ijet]["pt"]
                    end
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end
