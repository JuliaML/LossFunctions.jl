module LossFunctions

using UnicodePlots

import Base: show, call, print, transpose
import UnicodePlots: lineplot, lineplot!

export

    sigmoid,

    Cost,
      Loss,
        SupervisedLoss,
          MarginBasedLoss,
            LogitMarginLoss,
            HingeLoss,
            SqrHingeLoss,
            SqrSmoothedHingeLoss,
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
    issymmetric

include("common.jl")
include("abstract_cost.jl")
include("margin_based.jl")
include("distance_based.jl")
include("other.jl")
include("io.jl")

end # module
