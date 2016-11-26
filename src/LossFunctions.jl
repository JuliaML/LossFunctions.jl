__precompile__(true)

module LossFunctions

using RecipesBase

import Base.*
using Base.Cartesian

# to be replaced with Reexport as soon as it's importall issues are fixed
importall LearnBase
eval(Expr(:toplevel, Expr(:export, setdiff(names(LearnBase), [:LearnBase])...)))

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

    PoissonLoss,
    LogitProbLoss,
    CrossentropyLoss,
    ZeroOneLoss,

    ScaledLoss

include("common.jl")

include("supervised/supervised.jl")
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/scaledloss.jl")
include("supervised/other.jl")
include("supervised/io.jl")

# allow using SupervisedLoss as function
for T in filter(isleaftype,subtypes(SupervisedLoss))
    @eval (loss::$T)(target, output) = value(loss, target, output)
end

# allow using MarginLoss and DistanceLoss as function
for T in union(subtypes(DistanceLoss), subtypes(MarginLoss))
    @eval (loss::$T)(target, output) = value(loss, target, output)
    @eval (loss::$T)(x) = value(loss, x)
end

end # module
