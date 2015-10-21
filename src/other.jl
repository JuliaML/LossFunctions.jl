
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

# ==========================================================================
# L(y, t) = sign(yt) < 0 ? 1 : 0

immutable ZeroOneLoss <: SupervisedLoss end

call(l::ZeroOneLoss, yt::Real) = value(l, yt)
transpose(l::ZeroOneLoss) = repr_deriv_fun(l)
value(l::ZeroOneLoss, y::Real, t::Real) = value(l, y * t)
deriv(l::ZeroOneLoss, y::Real, t::Real) = zero(t)
deriv2(l::ZeroOneLoss, y::Real, t::Real) = zero(t)

value{T<:Real}(l::ZeroOneLoss, yt::T) = sign(yt) < 0 ? one(T) : zero(T)
deriv{T<:Real}(l::ZeroOneLoss, yt::T) = zero(T)
deriv2{T<:Real}(l::ZeroOneLoss, yt::T) = zero(T)

function repr_fun(l::ZeroOneLoss)
  _φ(yt::Real) = value(l, yt)
  _φ
end

function repr_deriv_fun(l::ZeroOneLoss)
  _φ_deriv(yt::Real) = deriv(l, yt)
  _φ_deriv
end

function repr_deriv2_fun(l::ZeroOneLoss)
  _φ_deriv2(yt::Real) = deriv2(l, yt)
  _φ_deriv2
end

isdifferentiable(::ZeroOneLoss) = false
isconvex(::ZeroOneLoss) = false
isclasscalibrated(l::ZeroOneLoss) = true
