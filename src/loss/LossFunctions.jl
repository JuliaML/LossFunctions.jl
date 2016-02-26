

include("margin_based.jl")
include("distance_based.jl")
include("other.jl")
include("io.jl")

@autocomplete LossFunctions export

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
    