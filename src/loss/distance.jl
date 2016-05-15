# ==========================================================================
# L(y, t) = |y - t|^P

immutable LPDistLoss{P} <: DistanceLoss
    LPDistLoss() = typeof(P) <: Number ? new() : error()
end

LPDistLoss(p::Number) = LPDistLoss{p}()

value{P}(loss::LPDistLoss{P}, residual::Number) = abs(residual)^P
function deriv{P,T<:Number}(loss::LPDistLoss{P}, residual::T)
    if residual == 0
        zero(residual)
    else
        P * residual * abs(residual)^(P-convert(typeof(P), 2))
    end
end
function deriv2{P,T<:Number}(loss::LPDistLoss{P}, residual::T)
    if residual == 0
        zero(residual)
    else
        (abs2(P)-P) * abs(residual)^P / abs2(residual)
    end
end
value_deriv{P}(loss::LPDistLoss{P}, residual::Number) = (value(loss,residual), deriv(loss,residual))

issymmetric{P}(::LPDistLoss{P}) = true
isdifferentiable{P}(::LPDistLoss{P}) = P > 1
isdifferentiable{P}(::LPDistLoss{P}, at) = P > 1 || at != 0
istwicedifferentiable{P}(::LPDistLoss{P}) = P > 1
istwicedifferentiable{P}(::LPDistLoss{P}, at) = P > 1 || at != 0
islipschitzcont{P}(::LPDistLoss{P}) = P == 1
islipschitzcont_deriv{P}(::LPDistLoss{P}) = 1 <= P <= 2
isconvex{P}(::LPDistLoss{P}) = P >= 1
isstronglyconvex{P}(::LPDistLoss{P}) = P > 1

# ==========================================================================
# L(y, t) = |y - t|

typealias L1DistLoss LPDistLoss{1}

sumvalue(loss::L1DistLoss, residual::AbstractArray) = sumabs(residual)
value(loss::L1DistLoss, residual::Number) = abs(residual)
deriv{T<:Number}(loss::L1DistLoss, residual::T) = convert(T, sign(residual))
deriv2{T<:Number}(loss::L1DistLoss, residual::T) = zero(T)
value_deriv(loss::L1DistLoss, residual::Number) = (abs(residual), sign(residual))

isdifferentiable(::L1DistLoss) = false
isdifferentiable(::L1DistLoss, at) = at != 0
istwicedifferentiable(::L1DistLoss) = true
istwicedifferentiable(::L1DistLoss, at) = true
islipschitzcont(::L1DistLoss) = true
islipschitzcont_deriv(::L1DistLoss) = true
isconvex(::L1DistLoss) = true
isstronglyconvex(::L1DistLoss) = false

# ==========================================================================
# L(y, t) = (y - t)^2

typealias L2DistLoss LPDistLoss{2}

sumvalue(loss::L2DistLoss, residual::AbstractArray) = sumabs2(residual)
value(loss::L2DistLoss, residual::Number) = abs2(residual)
deriv{T<:Number}(loss::L2DistLoss, residual::T) = T(2) * residual
deriv2{T<:Number}(loss::L2DistLoss, residual::T) = T(2)
value_deriv{T<:Number}(loss::L2DistLoss, residual::T) = (abs2(residual), T(2) * residual)

isdifferentiable(::L2DistLoss) = true
isdifferentiable(::L2DistLoss, at) = true
istwicedifferentiable(::L2DistLoss) = true
istwicedifferentiable(::L2DistLoss, at) = true
islipschitzcont(::L2DistLoss) = false
islipschitzcont_deriv(::L2DistLoss) = true
isconvex(::L2DistLoss) = true
isstronglyconvex(::L2DistLoss) = true

# ==========================================================================
# L(y, t) = max(0, |y - t| - ɛ)

immutable L1EpsilonInsLoss <: DistanceLoss
    eps::Float64

    function L1EpsilonInsLoss(ɛ::Number)
        ɛ > 0 || error("ɛ must be strictly positive")
        new(convert(Float64, ɛ))
    end
end
typealias EpsilonInsLoss L1EpsilonInsLoss

value{T<:Number}(loss::L1EpsilonInsLoss, residual::T) = max(zero(T), abs(residual) - loss.eps)
deriv{T<:Number}(loss::L1EpsilonInsLoss, residual::T) = abs(residual) <= loss.eps ? zero(T) : sign(residual)
deriv2{T<:Number}(loss::L1EpsilonInsLoss, residual::T) = zero(T)
function value_deriv{T<:Number}(loss::L1EpsilonInsLoss, residual::T)
    absr = abs(residual)
    absr <= loss.eps ? (zero(T), zero(T)) : (absr - loss.eps, sign(residual))
end

issymmetric(::L1EpsilonInsLoss) = true
isdifferentiable(::L1EpsilonInsLoss) = false
isdifferentiable(loss::L1EpsilonInsLoss, at) = abs(at) != loss.eps
istwicedifferentiable(::L1EpsilonInsLoss) = true
istwicedifferentiable(loss::L1EpsilonInsLoss, at) = abs(at) != loss.eps
islipschitzcont(::L1EpsilonInsLoss) = true
islipschitzcont_deriv(::L1EpsilonInsLoss) = true
isconvex(::L1EpsilonInsLoss) = true
isstronglyconvex(::L1EpsilonInsLoss) = false

# ==========================================================================
# L(y, t) = max(0, |y - t| - ɛ)^2

immutable L2EpsilonInsLoss <: DistanceLoss
    eps::Float64

    function L2EpsilonInsLoss(ɛ::Number)
        ɛ > 0 || error("ɛ must be strictly positive")
        new(convert(Float64, ɛ))
    end
end

value{T<:Number}(loss::L2EpsilonInsLoss, residual::T) = abs2(max(zero(T), abs(residual) - loss.eps))
function deriv{T<:Number}(loss::L2EpsilonInsLoss, residual::T)
    absr = abs(residual)
    absr <= loss.eps ? zero(T) : T(2)*sign(residual)*(absr - loss.eps)
end
deriv2{T<:Number}(loss::L2EpsilonInsLoss, residual::T) = abs(residual) <= loss.eps ? zero(T) : T(2)
function value_deriv{T<:Number}(loss::L2EpsilonInsLoss, residual::T)
    absr = abs(residual)
    diff = absr - loss.eps
    absr <= loss.eps ? (zero(T), zero(T)) : (abs2(diff), T(2)*sign(residual)*diff)
end

issymmetric(::L2EpsilonInsLoss) = true
isdifferentiable(::L2EpsilonInsLoss) = true
isdifferentiable(::L2EpsilonInsLoss, at) = true
istwicedifferentiable(::L2EpsilonInsLoss) = false
istwicedifferentiable(loss::L2EpsilonInsLoss, at) = abs(at) != loss.eps
islipschitzcont(::L2EpsilonInsLoss) = true
islipschitzcont_deriv(::L2EpsilonInsLoss) = true
isconvex(::L2EpsilonInsLoss) = true
isstronglyconvex(::L2EpsilonInsLoss) = true

# ==========================================================================
# L(y, t) = -ln(4 * exp(y - t) / (1 + exp(y - t))²)

immutable LogitDistLoss <: DistanceLoss end

function value(loss::LogitDistLoss, residual::Number)
    er = exp(residual)
    T = typeof(er)
    -log(T(4) * er / abs2(one(T) + er))
end
function deriv{T<:Number}(loss::LogitDistLoss, residual::T)
    tanh(residual / T(2))
end
function deriv2(loss::LogitDistLoss, residual::Number)
    er = exp(residual)
    T = typeof(er)
    T(2)*er / abs2(one(T) + er)
end
function value_deriv(loss::LogitDistLoss, residual::Number)
    er = exp(residual)
    T = typeof(er)
    er1 = one(T) + er
    -log(T(4) * er / abs2(er1)), (er - one(T)) / (er1)
end

issymmetric(::LogitDistLoss) = true
isdifferentiable(::LogitDistLoss) = true
isdifferentiable(::LogitDistLoss, at) = true
istwicedifferentiable(::LogitDistLoss) = true
istwicedifferentiable(::LogitDistLoss, at) = true
islipschitzcont(::LogitDistLoss) = true
islipschitzcont_deriv(::LogitDistLoss) = true
isconvex(::LogitDistLoss) = true
isstronglyconvex(::LogitDistLoss) = true

