module Penalties

import ..LearnBase: value, value!, deriv, deriv!, deriv2, value_deriv, grad, grad!,
                    value_fun, deriv_fun, deriv2_fun, value_deriv_fun,
                    addgrad!
import ..LearnBase: isminimizable, isdifferentiable, istwicedifferentiable,
                    isconvex, islipschitzcont, islocallylipschitzcont,
                    isclipable, ismarginbased, issymmetric
import ..LearnBase: Penalty
import ..LearnBase: @_dimcheck
import Base: show, call, print, transpose

export

    NoPenalty,
    L1Penalty,
    L2Penalty
    # ElasticNetPenalty,
    # SCADPenalty,

include("penalty.jl")

end
