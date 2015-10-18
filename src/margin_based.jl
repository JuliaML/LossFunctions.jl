
# ==========================================================================
# L(y, t) = ln(1 + exp(-yt))

immutable LogitLoss <: MarginBasedLoss end

logit_loss(y::Real, t::Real) = logit_loss(y * t)
logit_loss(yt::Real) = log1p(exp(-yt))

logit_deriv(y::Real, t::Real) = logit_deriv(y * t)
function logit_deriv(yt::Real)
  eᵗ = exp(-yt)
  -eᵗ / (one(eᵗ) + eᵗ)
end

logit_deriv2(y::Real, t::Real) = logit_deriv2(y * t)
function logit_deriv2(yt::Real)
  eᵗ = exp(-yt)
  eᵗ / abs2(one(eᵗ) + eᵗ)
end

logit_loss_deriv(y::Real, t::Real) = logit_loss_deriv(y * t)
function logit_loss_deriv(yt::Real)
  eᵗ = exp(-yt)
  log1p(eᵗ), -eᵗ / (one(eᵗ) + eᵗ)
end

value(l::LogitLoss, yt::Real) = logit_loss(yt)
deriv(l::LogitLoss, yt::Real) = logit_deriv(yt)
deriv2(l::LogitLoss, yt::Real) = logit_deriv2(yt)
value_deriv(l::LogitLoss, yt::Real) = logit_loss_deriv(yt)

isunivfishercons(::LogitLoss) = true
isdifferentiable(::LogitLoss) = true
islipschitzcont(::LogitLoss) = true
isconvex(::LogitLoss) = true
isclipable(::LogitLoss) = true
