module LossFunctions

import Base.*
using Base.Cartesian
using Markdown
using SparseArrays
using RecipesBase

import LearnBase.AggMode
import LearnBase.ObsDim
import LearnBase:
    AggregateMode,
    Loss, SupervisedLoss,
    DistanceLoss, MarginLoss,
    value, value!,
    deriv, deriv!,
    deriv2, deriv2!,
    isdistancebased, ismarginbased,
    isminimizable, isdifferentiable,
    istwicedifferentiable,
    isconvex, isstrictlyconvex,
    isstronglyconvex, isnemitski,
    isunivfishercons, isfishercons,
    islipschitzcont, islocallylipschitzcont,
    isclipable, isclasscalibrated, issymmetric

# supervised losses
include("supervised.jl")

# IO and plot recipes
include("printing.jl")
include("plotrecipes.jl")

# deprecations
@deprecate LogitProbLoss CrossEntropyLoss
@deprecate PinballLoss QuantileLoss
@deprecate OrdinalHingeLoss OrdinalMarginLoss{HingeLoss}
@deprecate ScaledDistanceLoss ScaledLoss
@deprecate ScaledMarginLoss ScaledLoss
@deprecate weightedloss(l, w) WeightedMarginLoss(l, w)
@deprecate scaled(l, λ) ScaledLoss(l, λ)
@deprecate value_deriv(l,y,ŷ) (value(l,y,ŷ), deriv(l,y,ŷ))

export
    # loss API
    Loss,
    SupervisedLoss,
    MarginLoss,
    DistanceLoss,
    value, value!,
    deriv, deriv!,
    deriv2, deriv2!,
    isdistancebased, ismarginbased,
    isminimizable, isdifferentiable,
    istwicedifferentiable,
    isconvex, isstrictlyconvex,
    isstronglyconvex, isnemitski,
    isunivfishercons, isfishercons,
    islipschitzcont, islocallylipschitzcont,
    isclipable, isclasscalibrated, issymmetric,

    # relevant submodules
    AggMode, ObsDim,

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

    # other losses
    MisclassLoss,
    PoissonLoss,
    CrossEntropyLoss,

    # meta losses
    ScaledLoss,
    OrdinalMarginLoss,
    WeightedMarginLoss

end # module
