__precompile__()

module Losses

importall LearnBase
using RecipesBase

export

    value_fun,
    deriv_fun,
    deriv2_fun,
    grad_fun,
    value_deriv_fun,
    value_grad_fun,

    LogitMarginLoss,
    PerceptronLoss,
    HingeLoss,
    L1HingeLoss,
    L2HingeLoss,
    SmoothedL1HingeLoss,
    ModifiedHuberLoss,

    LPDistLoss,
    L1DistLoss,
    L2DistLoss,
    EpsilonInsLoss,
    L1EpsilonInsLoss,
    L2EpsilonInsLoss,
    LogitDistLoss,

    LogitProbLoss,
    CrossentropyLoss,
    ZeroOneLoss

include("common.jl")

include("supervised/supervised.jl")
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/other.jl")
include("supervised/io.jl")

end # module
