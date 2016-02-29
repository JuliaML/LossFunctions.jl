module LearnBase

using Reexport
@reexport using MLBase
using Requires
using ArrayViews
# using UnicodePlots

import Base: show, shuffle!, convert, call, print, transpose, minimum, copy
@require UnicodePlots begin
    import UnicodePlots: lineplot, lineplot!
end
import StatsBase: fit, fit!, predict, predict!, nobs, coef,
                  deviance, loglikelihood, coeftable, stderr,
                  vcov, confint, residuals, model_response
import MLBase: labelencode, labeldecode, groupindices

export

    sigmoid,
    Predictor,
        LinearPredictor,
        SigmoidPredictor,

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
    issymmetric

include("common.jl")
include("encoding/encoding.jl")
include("loss/loss.jl")
include("risk/prediction.jl")
include("risk/empiricalrisk.jl")
include("risk/riskfunc.jl")
include("risk/riskmodel.jl")
include("optim/minimizeable.jl")
include("optim/optimize.jl")
include("optim/paramupdater.jl")

end # module
