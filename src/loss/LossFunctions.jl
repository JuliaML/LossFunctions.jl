module LossFunctions

using UnicodePlots
import UnicodePlots: lineplot, lineplot!
import ..LearnBase: value, value!, deriv, deriv!, deriv2, value_deriv, grad, grad!,
                    sumvalue, sumderiv, meanvalue, meanderiv,
                    value_fun, deriv_fun, deriv2_fun, value_deriv_fun
import ..LearnBase: isminimizable, isdifferentiable, istwicedifferentiable,
                    isconvex, isnemitski, islipschitzcont, islocallylipschitzcont,
                    isclipable, ismarginbased, isclasscalibrated, isdistancebased,
                    issymmetric, isfishercons, isunivfishercons
import ..LearnBase: Cost, Loss, SupervisedLoss, UnsupervisedLoss,
                    MarginBasedLoss, DistanceBasedLoss
import ..LearnBase: @_dimcheck
import Base: show, call, print, transpose, copy

export

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

include("margin_based.jl")
include("distance_based.jl")
include("other.jl")
include("io.jl")

end
