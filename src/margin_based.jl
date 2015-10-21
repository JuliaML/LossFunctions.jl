
# ==========================================================================
# L(y, t) = ln(1 + exp(-yt))

immutable LogitMarginLoss <: MarginBasedLoss end

value(l::LogitMarginLoss, yt::Real) = log1p(exp(-yt))
deriv(l::LogitMarginLoss, yt::Real) = (eᵗ = exp(-yt); -eᵗ / (1 + eᵗ))
deriv2(l::LogitMarginLoss, yt::Real) = (eᵗ = exp(-yt); eᵗ / abs2(1 + eᵗ))
value_deriv(l::LogitMarginLoss, yt::Real) = (eᵗ = exp(-yt); (log1p(eᵗ), -eᵗ / (1 + eᵗ)))

isunivfishercons(::LogitMarginLoss) = true
isdifferentiable(::LogitMarginLoss) = true
isdifferentiable(::LogitMarginLoss, at) = true
islipschitzcont(::LogitMarginLoss) = true
isconvex(::LogitMarginLoss) = true
isclipable(::LogitMarginLoss) = false

# ==========================================================================
# L(y, t) = max(0, 1 - yt)

immutable HingeLoss <: MarginBasedLoss end

value{T<:Real}(l::HingeLoss, yt::T) = max(zero(T), 1 - yt)
deriv{T<:Real}(l::HingeLoss, yt::T) = yt >= 1 ? zero(T) : -one(T)
deriv2{T<:Real}(l::HingeLoss, yt::T) = zero(T)
value_deriv{T<:Real}(l::HingeLoss, yt::T) = yt >= 1 ? (zero(T), zero(T)) : (1 - yt, -one(T))

isdifferentiable(::HingeLoss) = false
isdifferentiable(::HingeLoss, at) = at != 1
islipschitzcont(::HingeLoss) = true
isconvex(::HingeLoss) = true
isclipable(::HingeLoss) = true

# ==========================================================================
# L(y, t) = max(0, 1 - yt)^2

immutable SqrHingeLoss <: MarginBasedLoss end

value{T<:Real}(l::SqrHingeLoss, yt::T) = yt >= 1 ? zero(T) : abs2(1 - yt)
deriv{T<:Real}(l::SqrHingeLoss, yt::T) = yt >= 1 ? zero(T) : 2(yt - one(T))
deriv2{T<:Real}(l::SqrHingeLoss, yt::T) = yt >= 1 ? zero(T) : convert(T, -2)
value_deriv{T<:Real}(l::SqrHingeLoss, yt::T) = yt >= 1 ? (zero(T), zero(T)) : (1 - yt, -one(T))

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
    yt >= 1 ? zero(T) : (yt - one(T)) / l.γ
  else
    -one(T)
  end
end

function deriv2(l::SqrSmoothedHingeLoss, yt::Real)
  yt < 1 - l.γ && yt >= 1 ? zero(T) : one(T)
end

value_deriv(l::SqrSmoothedHingeLoss, yt::Real) = (value(l, yt), deriv(l, yt))

isdifferentiable(::SqrSmoothedHingeLoss) = true
isdifferentiable(::SqrSmoothedHingeLoss, at) = true
islocallylipschitzcont(::SqrSmoothedHingeLoss) = true
isconvex(::SqrSmoothedHingeLoss) = true
isclipable(::SqrSmoothedHingeLoss) = true
