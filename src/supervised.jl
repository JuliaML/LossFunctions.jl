Base.Broadcast.broadcastable(l::SupervisedLoss) = Ref(l)

# --------------------------------------------------------------
# SupervisedLoss fallbacks

function value_deriv(l::SupervisedLoss, target::Number, output::Number)
    value(l, target, output), deriv(l, target, output)
end

isstronglyconvex(::SupervisedLoss) = false
isdifferentiable(l::SupervisedLoss) = istwicedifferentiable(l)
isdifferentiable(l::SupervisedLoss, at) = isdifferentiable(l)
istwicedifferentiable(::SupervisedLoss) = false
istwicedifferentiable(l::SupervisedLoss, at) = istwicedifferentiable(l)
islocallylipschitzcont(l::SupervisedLoss) = isconvex(l) || islipschitzcont(l)
islipschitzcont(::SupervisedLoss) = false
isnemitski(l::SupervisedLoss) = islocallylipschitzcont(l)
isclipable(::SupervisedLoss) = false
ismarginbased(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false
isminimizable(l::SupervisedLoss) = isconvex(l)
isclasscalibrated(::SupervisedLoss) = false

# --------------------------------------------------------------
# MarginLoss fallbacks

value(loss::MarginLoss, target::Number, output::Number) = value(loss, target * output)
deriv(loss::MarginLoss, target::Number, output::Number) = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)
function value_deriv(loss::MarginLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end
value_deriv(loss::MarginLoss, agreement::Number) = (value(loss, agreement), deriv(loss, agreement))

isunivfishercons(::MarginLoss) = false
isfishercons(loss::MarginLoss) = isunivfishercons(loss)
isnemitski(::MarginLoss) = true
ismarginbased(::MarginLoss) = true

# For every convex margin based loss L(a) the following statements
# are equivalent:
#   - L is classification calibrated
#   - L is differentiable at 0 and L'(0) < 0
# We use this here as fallback implementation.
# The other direction of the implication is not implemented however.
isclasscalibrated(loss::MarginLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

# --------------------------------------------------------------
# DistanceLoss fallbacks

value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)
value_deriv(loss::DistanceLoss, difference::Number) = (value(loss, difference), deriv(loss, difference))

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false
isclipable(::DistanceLoss) = true

# -----------------
# IMPLEMENTATIONS
# -----------------
include("supervised/sparse.jl")
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/scaled.jl")
include("supervised/weightedbinary.jl")
include("supervised/other.jl")
include("supervised/ordinal.jl")
