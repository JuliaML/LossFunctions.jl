module LossFunctions

using UnicodePlots

import Base: show, call, print, transpose
import UnicodePlots: lineplot, lineplot!

export

    Cost,
      Loss,
        SupervisedLoss,
          MarginBasedLoss,
            LogitLoss,
              logit_loss,
              logit_deriv,
              logit_deriv2,
              logit_loss_deriv,
            HingeLoss,
              hinge_loss,
              hinge_deriv,
              hinge_deriv2,
              hinge_loss_deriv,
            SqrHingeLoss,
              sqrhinge_loss,
              sqrhinge_deriv,
              sqrhinge_deriv2,
              sqrhinge_loss_deriv,
            SqrSmoothedHingeLoss,
          DistanceBasedLoss,
            LPLoss,
            L1Loss,
              l1_loss,
              l1_deriv,
              l1_deriv2,
              l1_loss_deriv,
            L2Loss,
              l2_loss,
              l2_deriv,
              l2_deriv2,
              l2_loss_deriv,
          CrossentropyLoss,
            crossentropy_loss,
            crossentropy_deriv,
            crossentropy_deriv2,
            crossentropy_loss_deriv,
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
