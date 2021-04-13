module OptimalPlasticityRatios

using StaticArrays
using Flux
using Random
using LinearAlgebra
using Statistics
using RecipesBase
using Tracker

include("miscellaneous.jl")
include("updateInfo.jl")
include("trainingInfo.jl")
include("lossInfo.jl")
include("recordingInfo.jl")
include("updateMethods.jl")
include("trainingLoop.jl")
include("dataParsing.jl")

export training_loop!, get_model, recorder, extract_endmean





end # module
