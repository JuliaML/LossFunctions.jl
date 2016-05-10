
__precompile__()

module LearnBase

using Reexport
@reexport using MLBase
using Requires
using RecipesBase

import Base: show, shuffle!, convert, call, print, transpose, minimum, copy
# @require UnicodePlots begin
#     import UnicodePlots: lineplot, lineplot!
# end
import StatsBase: fit, fit!, predict, predict!, nobs, coef,
                  deviance, loglikelihood, coeftable, stderr,
                  vcov, confint, residuals, model_response
import MLBase: labelencode, labeldecode, groupindices

export

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
include("transform/transform.jl")
include("risk/risk.jl")
include("optim/optim.jl")

end # module
