
# ==========================================================================
# L(y, t) = - y*ln(t) - (1-y)*ln(1-t)

immutable CrossentropyLoss <: SupervisedLoss end
typealias LogitProbLoss CrossentropyLoss

function value(l::CrossentropyLoss, y::Real, t::Real)
  if y == 1
    -log(t)
  elseif y == 0
    -log(1 - t)
  else
    -y*log(t) - (1-y)*log(1-t)
  end  
end
deriv(l::CrossentropyLoss, y::Real, t::Real) = t - y
deriv2(l::CrossentropyLoss, y::Real, t::Real) = 1
value_deriv(l::CrossentropyLoss, y::Real, t::Real) = (value(l,y,t), deriv(l,y,t))

isdifferentiable(::CrossentropyLoss) = true
isconvex(::CrossentropyLoss) = true
