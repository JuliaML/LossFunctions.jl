__precompile__()

module Losses

using RecipesBase

import Base.*

# to be replaced with Reexport as soon as it's importall issues are fixed
importall LearnBase
eval(Expr(:toplevel, Expr(:export, setdiff(names(LearnBase), [:LearnBase])...)))
using Compat

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
    PeriodicLoss,
    HuberLoss,
    EpsilonInsLoss,
    L1EpsilonInsLoss,
    L2EpsilonInsLoss,
    LogitDistLoss,

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


end # module
