
# ==========================================================================
# L(y, t) = - y*ln(t) - (1-y)*ln(1-t)

immutable CrossentropyLoss <: SupervisedLoss end

crossentropy_loss(y::Real, t::Real) = y > 0 ? -log(t) : -log(1 - t)
crossentropy_deriv(y::Real, t::Real) = t - y
crossentropy_deriv2(y::Real, t::Real) = 1
crossentropy_loss_deriv(y::Real, t::Real) = y > 0 ? (-log(t), t - 1) : (-log(1 - t), t)

value(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_loss(y, t)
deriv(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_deriv(y, t)
deriv2(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_deriv2(y, t)
value_deriv(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_loss_deriv(y, t)

isdifferentiable(::CrossentropyLoss) = true
isconvex(::CrossentropyLoss) = true
