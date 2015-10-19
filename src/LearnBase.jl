module LearnBase

using Reexport
@reexport using StatsBase
@reexport using MLBase
using ArrayViews

import Base: show, shuffle!, convert
import StatsBase: fit, fit!, predict, predict!, nobs, coef, 
                  deviance, loglikelihood, coeftable, stderr,
                  vcov, confint, residuals, model_response
import MLBase: labelencode, labeldecode, groupindices

export

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

end # module
