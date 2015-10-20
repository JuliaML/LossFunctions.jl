
# ==========================================================================
# L(y, t) = ln(1 + exp(-yt))

immutable LogitLoss <: MarginBasedLoss end

logit_loss(y::Real, t::Real) = logit_loss(y * t)
logit_loss(yt::Real) = log1p(exp(-yt))

logit_deriv(y::Real, t::Real) = logit_deriv(y * t)
function logit_deriv(yt::Real)
  eᵗ = exp(-yt)
  -eᵗ / (1 + eᵗ)
end

logit_deriv2(y::Real, t::Real) = logit_deriv2(y * t)
function logit_deriv2(yt::Real)
  eᵗ = exp(-yt)
  eᵗ / abs2(1 + eᵗ)
end

logit_loss_deriv(y::Real, t::Real) = logit_loss_deriv(y * t)
function logit_loss_deriv(yt::Real)
  eᵗ = exp(-yt)
  log1p(eᵗ), -eᵗ / (1 + eᵗ)
end

value(l::LogitLoss, yt::Real) = logit_loss(yt)
deriv(l::LogitLoss, yt::Real) = logit_deriv(yt)
deriv2(l::LogitLoss, yt::Real) = logit_deriv2(yt)
value_deriv(l::LogitLoss, yt::Real) = logit_loss_deriv(yt)

isunivfishercons(::LogitLoss) = true
isdifferentiable(::LogitLoss) = true
isdifferentiable(::LogitLoss, at) = true
islipschitzcont(::LogitLoss) = true
isconvex(::LogitLoss) = true
isclipable(::LogitLoss) = true

# ==========================================================================
# L(y, t) = max(0, 1 - yt)

immutable HingeLoss <: MarginBasedLoss end

hinge_loss(y::Real, t::Real) = hinge_loss(y * t)
hinge_loss{T<:Real}(yt::T) = max(zero(T), 1 - yt)

hinge_deriv(y::Real, t::Real) = hinge_deriv(y * t)
function hinge_deriv{T<:Real}(yt::T)
  yt > 1 ? zero(T) : -one(T)
end

hinge_deriv2{T<:Real}(y::Real, t::T) = zero(T)
hinge_deriv2{T<:Real}(yt::T) = zero(T)

hinge_loss_deriv(y::Real, t::Real) = hinge_loss_deriv(y * t)
function hinge_loss_deriv{T<:Real}(yt::T)
  yt > 1 ? (zero(T), zero(T)) : (1 - yt, -one(T))
end

value(l::HingeLoss, yt::Real) = hinge_loss(yt)
deriv(l::HingeLoss, yt::Real) = hinge_deriv(yt)
deriv2(l::HingeLoss, yt::Real) = hinge_deriv2(yt)
value_deriv(l::HingeLoss, yt::Real) = hinge_loss_deriv(yt)

isdifferentiable(::HingeLoss) = false
isdifferentiable(::HingeLoss, at) = at != 1
islipschitzcont(::HingeLoss) = true
isconvex(::HingeLoss) = true
isclipable(::HingeLoss) = true

# ==========================================================================
# L(y, t) = max(0, 1 - yt)^2

immutable SqrHingeLoss <: MarginBasedLoss end

sqrhinge_loss(y::Real, t::Real) = sqrhinge_loss(y * t)
sqrhinge_loss{T<:Real}(yt::T) = yt > 1 ? zero(T) : abs2(1 - yt)

sqrhinge_deriv(y::Real, t::Real) = sqrhinge_deriv(y * t)
function sqrhinge_deriv{T<:Real}(yt::T)
  yt > 1 ? zero(T) : 2(yt - one(T))
end

sqrhinge_deriv2{T<:Real}(y::Real, t::T) = sqrhinge_deriv2(y * t)
sqrhinge_deriv2{T<:Real}(yt::T) = yt > 1 ? zero(T) : convert(T, -2)

sqrhinge_loss_deriv(y::Real, t::Real) = sqrhinge_loss_deriv(y * t)
function sqrhinge_loss_deriv{T<:Real}(yt::T)
  yt > 1 ? (zero(T), zero(T)) : (1 - yt, -one(T))
end

value(l::SqrHingeLoss, yt::Real) = sqrhinge_loss(yt)
deriv(l::SqrHingeLoss, yt::Real) = sqrhinge_deriv(yt)
deriv2(l::SqrHingeLoss, yt::Real) = sqrhinge_deriv2(yt)
value_deriv(l::SqrHingeLoss, yt::Real) = sqrhinge_loss_deriv(yt)

isdifferentiable(::SqrHingeLoss) = true
isdifferentiable(::SqrHingeLoss, at) = true
islocallylipschitzcont(::SqrHingeLoss) = true
isconvex(::SqrHingeLoss) = true
isclipable(::SqrHingeLoss) = true

# ==========================================================================
# L(y, t) = 0.5 / γ * max(0, 1 - yt)^2   ... yt >= 1 - γ
#           1 - γ / 2 - yt               ... otherwise

immutable SqrSmoothedHingeLoss <: MarginBasedLoss 
  γ::Float64

  function SqrSmoothedHingeLoss(γ::Real)
    γ > 0 || error("γ must be strictly positive")
    new(convert(Float64, γ))
  end
end

function value{T<:Real}(l::SqrSmoothedHingeLoss, yt::T)
  yt >= 1 - l.γ ? 0.5 / l.γ * abs2(max(zero(T), 1 - yt)) : one(T) - l.γ / 2 - yt
end

function deriv{T<:Real}(l::SqrSmoothedHingeLoss, yt::T)
  if yt >= 1 - l.γ
    yt > 1 ? zero(T) : (yt - one(T)) / l.γ
  else
    -one(T)
  end
end

deriv2(l::SqrSmoothedHingeLoss, yt::Real) = sqrhinge_deriv2(yt)
value_deriv(l::SqrSmoothedHingeLoss, yt::Real) = sqrhinge_loss_deriv(yt)
