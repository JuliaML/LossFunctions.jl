
# ==========================================================================
# L(y, t) = - y*ln(t) - (1-y)*ln(1-t)

immutable CrossentropyLoss <: SupervisedLoss end

function crossentropy_loss(y::Real, t::Real)
  if y == 1
    -log(t)
  elseif y == 0
    -log(1 - t)
  else
    -y*log(t)-(1-y)*log(1-t)
  end
end
crossentropy_deriv(y::Real, t::Real) = t - y
crossentropy_deriv2(y::Real, t::Real) = 1
crossentropy_loss_deriv(y::Real, t::Real) = (crossentropy_loss(y,t), crossentropy_deriv(y,t))

value(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_loss(y, t)
deriv(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_deriv(y, t)
deriv2(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_deriv2(y, t)
value_deriv(l::CrossentropyLoss, y::Real, t::Real) = crossentropy_loss_deriv(y, t)

isdifferentiable(::CrossentropyLoss) = true
isconvex(::CrossentropyLoss) = true
