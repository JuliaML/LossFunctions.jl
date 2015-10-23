module LearnBase

using Reexport
@reexport using StatsBase
@reexport using MLBase
using ArrayViews
using UnicodePlots

using UnicodePlots

import Base: show, shuffle!, convert, call, print, transpose
import UnicodePlots: lineplot, lineplot!
import StatsBase: fit, fit!, predict, predict!, nobs, coef, 
                  deviance, loglikelihood, coeftable, stderr,
                  vcov, confint, residuals, model_response
import MLBase: labelencode, labeldecode, groupindices

export

    sigmoid,

    Cost,
      Loss,
        SupervisedLoss,
          MarginBasedLoss,
            LogitMarginLoss,
            PerceptronLoss,
            HingeLoss,
            L1HingeLoss,
            L2HingeLoss,
            SmoothedL1HingeLoss,
          DistanceBasedLoss,
            LPDistLoss,
            L1DistLoss,
            L2DistLoss,
            EpsilonInsLoss,
            LogitDistLoss,
          LogitProbLoss,
          CrossentropyLoss,
          ZeroOneLoss,
        UnsupervisedLoss,

    value,
    deriv,
    deriv2,
    value_deriv,

    value_fun,
    deriv_fun,
    deriv2_fun,
    value_deriv_fun,
    repr_fun,
    repr_deriv_fun,
    repr_deriv2_fun,

    isminimizable,
    isdifferentiable,
    istwicedifferentiable,
    isconvex,
    isnemitski,
    isunivfishercons,
    isfishercons,
    islipschitzcont,
    islocallylipschitzcont,
    isclipable,
    ismarginbased,
    isclasscalibrated,
    isdistancebased,
    issymmetric,

    ClassEncoding,
      SignedClassEncoding,
      BinaryClassEncoding,
      MultinomialClassEncoding,
      OneHotClassEncoding,

    nclasses,
    labels,
    classdistribution,

    EncodedStatisticalModel,
    EncodedRegressionModel

include("common.jl")
include("classencoding.jl")
include("encodedmodel.jl")

include("loss/abstract_cost.jl")
include("loss/margin_based.jl")
include("loss/distance_based.jl")
include("loss/other.jl")
include("loss/io.jl")

end # module
