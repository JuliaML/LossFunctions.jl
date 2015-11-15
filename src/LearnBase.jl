module LearnBase

using Reexport
@reexport using MLBase
using ArrayViews
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
                DistanceBasedLoss,
            UnsupervisedLoss,

    LossFunctions,

    sigmoid,
    Predictor,
        LinearPredictor,
        SigmoidPredictor,

    Penalty,
    Penalties,

    RiskModel,

    AbstractSolver,

    RegressionData,

    value,
    value!,
    meanvalue,
    sumvalue,
    meanderiv,
    sumderiv,
    deriv,
    deriv!,
    grad,
    grad!,
    addgrad!,
    deriv2,
    value_deriv,

    value_fun,
    deriv_fun,
    deriv2_fun,
    value_deriv_fun,

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
include("data/regression.jl")
include("abstract_solver.jl")
include("abstract_cost.jl")
include("abstract_penalty.jl")
include("classencoding.jl")
include("encodedmodel.jl")
include("loss/LossFunctions.jl")
include("penalty/Penalties.jl")
include("risk/prediction.jl")
include("risk/riskspec.jl")

end # module
