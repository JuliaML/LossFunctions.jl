module LearnBase

using Reexport
@reexport using MLBase
using ArrayViews
using UnicodePlots
using Optim

import Base: show, shuffle!, convert, call, print, transpose, minimum
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

    EmpiricalRisk,
        EmpiricalRiskClassifier,
        EmpiricalRiskRegressor,
    RiskFunctional,

    AbstractSolver,

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
    value_grad,

    value_fun,
    deriv_fun,
    deriv2_fun,
    grad_fun,
    value_deriv_fun,
    value_grad_fun,

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
include("abstract_solver.jl")
include("abstract_cost.jl")
include("abstract_penalty.jl")
include("classencoding.jl")
include("encodedmodel.jl")
include("loss/LossFunctions.jl")
include("penalty/Penalties.jl")
include("risk/prediction.jl")
include("risk/empiricalrisk.jl")
include("risk/riskfunc.jl")
include("risk/riskmodel.jl")

end # module
