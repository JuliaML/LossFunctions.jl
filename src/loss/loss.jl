
include("abstract.jl")

export

    # autocomplete objects
    LossFunctions,
    ParameterLosses

# ------------------------------------
# Prediction Losses
# ------------------------------------

include("margin.jl")
include("distance.jl")
include("other.jl")

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


# ------------------------------------
# Parameter Losses
# ------------------------------------

include("params.jl")

@autocomplete ParameterLosses export
    NoParameterLoss,
    L1ParameterLoss,
    L2ParameterLoss
    # ElasticNetParameterLoss,
    # SCADParameterLoss,

# ------------------------------------

include("io.jl")
