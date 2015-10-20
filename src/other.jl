
# ==========================================================================
# L(y, t) = - y*ln(t) - (1-y)*ln(1-t)

immutable SigmoidCrossentropyLoss <: SupervisedLoss end

sigm_crossentropy_loss(y::Real, t::Real) = y > 0 ? -log(t) : -log(1 - t)
sigm_crossentropy_deriv(y::Real, t::Real) = t - y
sigm_crossentropy_deriv2(y::Real, t::Real) = 1
sigm_crossentropy_loss_deriv(y::Real, t::Real) = y > 0 ? (-log(t), t - 1) : (-log(1 - t), t)

value(l::SigmoidCrossentropyLoss, y::Real, t::Real) = sigm_crossentropy_loss(y, t)
deriv(l::SigmoidCrossentropyLoss, y::Real, t::Real) = sigm_crossentropy_deriv(y, t)
deriv2(l::SigmoidCrossentropyLoss, y::Real, t::Real) = sigm_crossentropy_deriv2(y, t)
value_deriv(l::SigmoidCrossentropyLoss, y::Real, t::Real) = sigm_crossentropy_loss_deriv(y, t)

isdifferentiable(::SigmoidCrossentropyLoss) = true
isconvex(::SigmoidCrossentropyLoss) = true
