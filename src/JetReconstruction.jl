"""
Jet reconstruction (reclustering) in Julia.
"""
module JetReconstruction

# particle type definition
include("Particle.jl")
export energy, px, py, pz, pt, phi, mass, eta, kt, ϕ, η

# Philipp's pseudojet
include("Pseudojet.jl")
export PseudoJet, rap, phi, pt2

# Simple HepMC3 reader
include("HepMC3.jl")
export HepMC3

# Algorithmic part, simple sequential implementation
include("Algo.jl")
export sequential_jet_reconstruct, kt_algo, anti_kt_algo, anti_kt_algo_alt, cambridge_aachen_algo

# Algorithmic part, tiled reconstruction strategy
include("TiledAlgo.jl")
export tiled_jet_reconstruct

# jet serialisation (saving to file)
include("Serialize.jl")
export savejets, loadjets!, loadjets

# jet visualisation
include("JetVis.jl")
export jetsplot

# JSON results
include("JSONresults.jl")
export FinalJet, FinalJets, JSON3

# Strategy to be used
@enum JetRecoStrategy Best N2Plain N2Tiled
export JetRecoStrategy, Best, N2Plain, N2Tiled

end
