module LearnBase

using Reexport
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
            ModifiedHuberLoss,
          DistanceBasedLoss,
            LPDistLoss,
            L1DistLoss,
            L2DistLoss,
            EpsilonInsLoss,
            L1EpsilonInsLoss,
            L2EpsilonInsLoss,
            LogitDistLoss,
          LogitProbLoss,
          CrossentropyLoss,
          ZeroOneLoss,
        UnsupervisedLoss,

    sigmoid,
    Predictor,
      LinearPredictor,
      SigmoidPredictor,

    Penalty,
      NoPenalty,
      L1Penalty,
      L2Penalty,
      ElasticNetPenalty,
      SCADPenalty,

    AbstractSolver,

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

include("abstract.jl")
include("common.jl")
include("classencoding.jl")
include("encodedmodel.jl")

include("loss/abstract_cost.jl")
include("loss/margin_based.jl")
include("loss/distance_based.jl")
include("loss/other.jl")
include("loss/io.jl")
include("penalty/penalty.jl")
include("prediction/prediction.jl")
include("risk/riskspec.jl")

end # module
