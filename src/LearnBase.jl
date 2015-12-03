module LearnBase

using Reexport
@reexport using MLBase
using ArrayViews
using UnicodePlots

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

    AbstractOptimizer,
    AbstractSolver,

    MinimizableFunctor,
        DifferentiableFunctor,
            TwiceDifferentiableFunctor,

    optimize,

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
    hess!,
    addgrad!,
    deriv2,
    value_deriv,
    value_deriv!,
    value_grad,
    value_grad!,

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
    isstronglyconvex,
    isnemitski,
    isunivfishercons,
    isfishercons,
    islipschitzcont,
    islocallylipschitzcont,
    islipschitzcont_deriv,
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
include("optim/minimizeable.jl")
include("optim/optimize.jl")

end # module
