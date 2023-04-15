module LossFunctions

using Markdown
using CategoricalArrays: CategoricalValue
using RecipesBase

import Base: *

# aggregation mode
include("aggmode.jl")

# trait functions
include("traits.jl")

# loss functions
include("losses.jl")

# IO and plot recipes
include("printing.jl")
include("plotrecipes.jl")

export
    # trait functions
    Loss,
    SupervisedLoss,
    MarginLoss,
    DistanceLoss,
    value, deriv, deriv2,
    isdistancebased, ismarginbased,
    isminimizable, isdifferentiable,
    istwicedifferentiable,
    isconvex, isstrictlyconvex,
    isstronglyconvex, isnemitski,
    isunivfishercons, isfishercons,
    islipschitzcont, islocallylipschitzcont,
    isclipable, isclasscalibrated, issymmetric,

    # relevant submodules
    AggMode,

    # margin-based losses
    ZeroOneLoss,
    LogitMarginLoss,
    PerceptronLoss,
    HingeLoss,
    L1HingeLoss,
    L2HingeLoss,
    SmoothedL1HingeLoss,
    ModifiedHuberLoss,
    L2MarginLoss,
    ExpLoss,
    SigmoidLoss,
    DWDMarginLoss,

    # distance-based losses
    LPDistLoss,
    L1DistLoss,
    L2DistLoss,
    PeriodicLoss,
    HuberLoss,
    EpsilonInsLoss,
    L1EpsilonInsLoss,
    L2EpsilonInsLoss,
    LogitDistLoss,
    QuantileLoss,
    LogCoshLoss,

    # other losses
    MisclassLoss,
    PoissonLoss,
    CrossEntropyLoss,

    # meta losses
    ScaledLoss,
    WeightedMarginLoss

end # module
