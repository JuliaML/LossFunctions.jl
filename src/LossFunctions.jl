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
    deriv, deriv2, deriv!,
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
    islipschitzcont_deriv, # maybe overkill
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
    value,
    value!,
    deriv,
    deriv!,
    deriv2,
    deriv2!,

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
    OrdinalHingeLoss,

    weightedloss,

    AggMode

include("devutils.jl")
include("aggregatemode.jl")

include("supervised/supervised.jl")
include("supervised/sparse.jl")
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/scaledloss.jl")
include("supervised/weightedbinary.jl")
include("supervised/other.jl")
include("supervised/ordinal.jl")
include("supervised/io.jl")

end # module
