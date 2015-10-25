
# ==========================================================================
# L(y, t) = |y - t|^P

immutable LPDistLoss{P} <: DistanceBasedLoss
  LPDistLoss() = typeof(P) <: Number ? new() : error()
end

LPDistLoss(p::Number) = LPDistLoss{p}()

value{P}(l::LPDistLoss{P}, r::Number) = abs(r)^P
function deriv{P,T<:Number}(l::LPDistLoss{P}, r::T)
  if r == 0
    zero(r)
  else
    P * r * abs(r)^(P-2)
  end
end
function deriv2{P,T<:Number}(l::LPDistLoss{P}, r::T)
  if r == 0
    zero(r)
  else
    (P^2-P) * abs(r)^P / r^2
  end
end
value_deriv{P}(l::LPDistLoss{P}, r::Number) = (value(l,r), deriv(l,r))

issymmetric{P}(::LPDistLoss{P}) = true
isdifferentiable{P}(::LPDistLoss{P}) = P > 1
isdifferentiable{P}(::LPDistLoss{P}, at) = P > 1 || at != 0
istwicedifferentiable{P}(::LPDistLoss{P}) = P > 1
istwicedifferentiable{P}(::LPDistLoss{P}, at) = P > 1 || at != 0
islipschitzcont{P}(::LPDistLoss{P}) = P == 1
isconvex{P}(::LPDistLoss{P}) = P >= 1

# ==========================================================================
# L(y, t) = |y - t|

typealias L1DistLoss LPDistLoss{1}

value(l::L1DistLoss, r::Number) = abs(r)
deriv{T<:Number}(l::L1DistLoss, r::T) = convert(T, sign(r))
deriv2{T<:Number}(l::L1DistLoss, r::T) = zero(T)
value_deriv(l::L1DistLoss, r::Number) = (abs(r), sign(r))

isdifferentiable(::L1DistLoss) = false
isdifferentiable(::L1DistLoss, at) = at != 0
istwicedifferentiable(::L1DistLoss) = true
istwicedifferentiable(::L1DistLoss, at) = true
islipschitzcont(::L1DistLoss) = true
isconvex(::L1DistLoss) = true

# ==========================================================================
# L(y, t) = (y - t)^2

typealias L2DistLoss LPDistLoss{2}

value(l::L2DistLoss, r::Number) = abs2(r)
deriv(l::L2DistLoss, r::Number) = 2r
deriv2{T<:Number}(l::L2DistLoss, r::T) = 2*one(T)
value_deriv(l::L2DistLoss, r::Number) = (abs2(r), 2r)

isdifferentiable(::L2DistLoss) = true
isdifferentiable(::L2DistLoss, at) = true
istwicedifferentiable(::L2DistLoss) = true
istwicedifferentiable(::L2DistLoss, at) = true
islipschitzcont(::L2DistLoss) = false
isconvex(::L2DistLoss) = true

# ==========================================================================
# L(y, t) = max(0, |y - t| - ɛ)

immutable L1EpsilonInsLoss <: DistanceBasedLoss
  eps::Float64

  function L1EpsilonInsLoss(ɛ::Number)
    ɛ > 0 || error("ɛ must be strictly positive")
    new(convert(Float64, ɛ))
  end
end
typealias EpsilonInsLoss L1EpsilonInsLoss

value{T<:Number}(l::L1EpsilonInsLoss, r::T) = max(zero(T), abs(r) - l.eps)
deriv{T<:Number}(l::L1EpsilonInsLoss, r::T) = abs(r) <= l.eps ? zero(T) : sign(r)
deriv2{T<:Number}(l::L1EpsilonInsLoss, r::T) = zero(T)
function value_deriv{T<:Number}(l::L1EpsilonInsLoss, r::T)
  absr = abs(r)
  absr <= l.eps ? (zero(T), zero(T)) : (absr - l.eps, sign(r))
end

issymmetric(::L1EpsilonInsLoss) = true
isdifferentiable(::L1EpsilonInsLoss) = false
isdifferentiable(l::L1EpsilonInsLoss, at) = abs(at) != l.eps
istwicedifferentiable(::L1EpsilonInsLoss) = true
istwicedifferentiable(l::L1EpsilonInsLoss, at) = abs(at) != l.eps
isconvex(::L1EpsilonInsLoss) = true

# ==========================================================================
# L(y, t) = max(0, |y - t| - ɛ)^2

immutable L2EpsilonInsLoss <: DistanceBasedLoss
  eps::Float64

  function L2EpsilonInsLoss(ɛ::Number)
    ɛ > 0 || error("ɛ must be strictly positive")
    new(convert(Float64, ɛ))
  end
end

value{T<:Number}(l::L2EpsilonInsLoss, r::T) = abs2(max(zero(T), abs(r) - l.eps))
function deriv{T<:Number}(l::L2EpsilonInsLoss, r::T)
  absr = abs(r)
  absr <= l.eps ? zero(T) : 2*sign(r)*(absr - l.eps)
end
deriv2{T<:Number}(l::L2EpsilonInsLoss, r::T) = abs(r) <= l.eps ? zero(T) : 2*one(T)
function value_deriv{T<:Number}(l::L2EpsilonInsLoss, r::T)
  absr = abs(r)
  diff = absr - l.eps
  absr <= l.eps ? (zero(T), zero(T)) : (abs2(diff), 2*sign(r)*diff)
end

issymmetric(::L2EpsilonInsLoss) = true
isdifferentiable(::L2EpsilonInsLoss) = true
isdifferentiable(::L2EpsilonInsLoss, at) = true
istwicedifferentiable(::L2EpsilonInsLoss) = false
istwicedifferentiable(l::L2EpsilonInsLoss, at) = abs(at) != l.eps
isconvex(::L2EpsilonInsLoss) = true

# ==========================================================================
# L(y, t) = -ln(4 * exp(y - t) / (1 + exp(y - t))²)

immutable LogitDistLoss <: DistanceBasedLoss end

function value(l::LogitDistLoss, r::Number)
  er = exp(r)
  -log(4 * er / abs2(1 + er))
end

function deriv(l::LogitDistLoss, r::Number)
  tanh(r / 2)
end

function deriv2(l::LogitDistLoss, r::Number)
  er = exp(r)
  2*er / abs2(1 + er)
end

function value_deriv(l::LogitDistLoss, r::Number)
  er = exp(r)
  er1 = 1 + er
  -log(4 * er / abs2(er1)), (er - 1) / (er1)
end

issymmetric(::LogitDistLoss) = true
isdifferentiable(::LogitDistLoss) = true
isdifferentiable(::LogitDistLoss, at) = true
istwicedifferentiable(::LogitDistLoss) = true
istwicedifferentiable(::LogitDistLoss, at) = true
isconvex(::LogitDistLoss) = true
