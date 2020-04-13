module LossFunctions

import Base.*
using Base.Cartesian
using Markdown
using SparseArrays
using InteractiveUtils
using RecipesBase
using LearnBase

import LearnBase:
    value, value!,
    deriv, deriv!,
    deriv2, deriv2!,
    value_deriv,
    scaled,
    isminimizable,
    isdifferentiable,
    istwicedifferentiable,
    isconvex,
    isstrictlyconvex,
    isstronglyconvex,
    isnemitski,
    isunivfishercons,
    isfishercons,
    islipschitzcont,
    islocallylipschitzcont,
    isclipable,
    ismarginbased,
    isclasscalibrated,
    isdistancebased,
    issymmetric

export

    Loss,
    SupervisedLoss,
    MarginLoss,
    DistanceLoss,
    value, value!,
    deriv, deriv!,
    deriv2, deriv2!,

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
    PinballLoss,

    MisclassLoss,
    PoissonLoss,
    LogitProbLoss,
    CrossEntropyLoss,
    ZeroOneLoss,

    OrdinalMarginLoss,

    weightedloss,

    AggMode

# for maintainers
include("devutils.jl")

# loss and aggregation
include("aggmode.jl")
include("supervised.jl")

# IO functionality
include("printing.jl")
include("plotrecipes.jl")

# deprecations
@deprecate OrdinalHingeLoss OrdinalMarginLoss{HingeLoss}

end # module
