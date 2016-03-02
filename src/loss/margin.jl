
# Note: agreement = output * target
#       Agreement is high when output and target are the same sign and |output| is large.
#       It is an indication that the output represents the correct class in a margin-based model.

# ==========================================================================
# L(target, output) = max(0, -agreement)

immutable PerceptronLoss <: MarginBasedLoss end

value{T<:Number}(loss::PerceptronLoss, agreement::T) = max(zero(T), -agreement)
deriv{T<:Number}(loss::PerceptronLoss, agreement::T) = agreement >= 0 ? zero(T) : -one(T)
deriv2{T<:Number}(loss::PerceptronLoss, agreement::T) = zero(T)
value_deriv{T<:Number}(loss::PerceptronLoss, agreement::T) = agreement >= 0 ? (zero(T), zero(T)) : (-agreement, -one(T))

isdifferentiable(::PerceptronLoss) = false
isdifferentiable(::PerceptronLoss, at) = at != 0
istwicedifferentiable(::PerceptronLoss) = false
istwicedifferentiable(::PerceptronLoss, at) = at != 0
islipschitzcont(::PerceptronLoss) = true
islipschitzcont_deriv(::PerceptronLoss) = true
isconvex(::PerceptronLoss) = true
isstronglyconvex(::PerceptronLoss) = false
isclipable(::PerceptronLoss) = true

# ==========================================================================
# L(target, output) = ln(1 + exp(-agreement))

immutable LogitMarginLoss <: MarginBasedLoss end

value(loss::LogitMarginLoss, agreement::Number) = log1p(exp(-agreement))
deriv(loss::LogitMarginLoss, agreement::Number) = -one(agreement) / (one(agreement) + exp(agreement))
deriv2(loss::LogitMarginLoss, agreement::Number) = (eᵗ = exp(agreement); eᵗ / abs2(one(eᵗ) + eᵗ))
value_deriv(loss::LogitMarginLoss, agreement::Number) = (eᵗ = exp(-agreement); (log1p(eᵗ), -eᵗ / (one(eᵗ) + eᵗ)))

isunivfishercons(::LogitMarginLoss) = true
isdifferentiable(::LogitMarginLoss) = true
isdifferentiable(::LogitMarginLoss, at) = true
istwicedifferentiable(::LogitMarginLoss) = true
istwicedifferentiable(::LogitMarginLoss, at) = true
islipschitzcont(::LogitMarginLoss) = true
islipschitzcont_deriv(::LogitMarginLoss) = true
isconvex(::LogitMarginLoss) = true
isstronglyconvex(::LogitMarginLoss) = true
isclipable(::LogitMarginLoss) = false

# ==========================================================================
# L(target, output) = max(0, 1 - agreement)

immutable L1HingeLoss <: MarginBasedLoss end
typealias HingeLoss L1HingeLoss

value{T<:Number}(loss::L1HingeLoss, agreement::T) = max(zero(T), one(T) - agreement)
deriv{T<:Number}(loss::L1HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : -one(T)
deriv2{T<:Number}(loss::L1HingeLoss, agreement::T) = zero(T)
value_deriv{T<:Number}(loss::L1HingeLoss, agreement::T) = agreement >= 1 ? (zero(T), zero(T)) : (one(T) - agreement, -one(T))

isdifferentiable(::L1HingeLoss) = false
isdifferentiable(::L1HingeLoss, at) = at != 1
istwicedifferentiable(::L1HingeLoss) = false
istwicedifferentiable(::L1HingeLoss, at) = at != 1
islipschitzcont(::L1HingeLoss) = true
islipschitzcont_deriv(::L1HingeLoss) = true
isconvex(::L1HingeLoss) = true
isstronglyconvex(::L1HingeLoss) = false
isclipable(::L1HingeLoss) = true

# ==========================================================================
# L(target, output) = max(0, 1 - agreement)^2

immutable L2HingeLoss <: MarginBasedLoss end

value{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : abs2(one(T) - agreement)
deriv{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : T(2) * (agreement - one(T))
deriv2{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : T(2)
value_deriv{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? (zero(T), zero(T)) : (abs2(one(T) - agreement), T(2) * (agreement - one(T)))

isdifferentiable(::L2HingeLoss) = true
isdifferentiable(::L2HingeLoss, at) = true
istwicedifferentiable(::L2HingeLoss) = false
istwicedifferentiable(::L2HingeLoss, at) = at != 1
islocallylipschitzcont(::L2HingeLoss) = true
islipschitzcont(::L2HingeLoss) = false
islipschitzcont_deriv(::L2HingeLoss) = true
isconvex(::L2HingeLoss) = true
isstronglyconvex(::L2HingeLoss) = true
isclipable(::L2HingeLoss) = true

# ==========================================================================
# L(target, output) = 0.5 / γ * max(0, 1 - agreement)^2   ... agreement >= 1 - γ
#                     1 - γ / 2 - agreement               ... otherwise

immutable SmoothedL1HingeLoss <: MarginBasedLoss
    gamma::Float64

    function SmoothedL1HingeLoss(γ::Number)
        γ > 0 || error("γ must be strictly positive")
        new(convert(Float64, γ))
    end
end

function value{T<:Number}(loss::SmoothedL1HingeLoss, agreement::T)
    agreement >= 1 - loss.gamma ? 0.5 / loss.gamma * abs2(max(zero(T), one(T) - agreement)) : one(T) - loss.gamma / 2 - agreement
end
function deriv{T<:Number}(loss::SmoothedL1HingeLoss, agreement::T)
    if agreement >= 1 - loss.gamma
        agreement >= 1 ? zero(T) : (agreement - one(T)) / loss.gamma
    else
        -one(T)
    end
end
function deriv2{T<:Number}(loss::SmoothedL1HingeLoss, agreement::T)
    agreement < 1 - loss.gamma || agreement > 1 ? zero(T) : one(T) / loss.gamma
end
value_deriv(loss::SmoothedL1HingeLoss, agreement::Number) = (value(loss, agreement), deriv(loss, agreement))

isdifferentiable(::SmoothedL1HingeLoss) = true
isdifferentiable(::SmoothedL1HingeLoss, at) = true
istwicedifferentiable(::SmoothedL1HingeLoss) = false
istwicedifferentiable(loss::SmoothedL1HingeLoss, at) = at != 1 && at != 1 - loss.gamma
islocallylipschitzcont(::SmoothedL1HingeLoss) = true
islipschitzcont(::SmoothedL1HingeLoss) = true
islipschitzcont_deriv(::SmoothedL1HingeLoss) = true
isconvex(::SmoothedL1HingeLoss) = true
isstronglyconvex(::SmoothedL1HingeLoss) = false
isclipable(::SmoothedL1HingeLoss) = true

# ==========================================================================
# L(target, output) = max(0, 1 - agreement)^2    ... agreement >= -1
#                     -4*agreement               ... otherwise

immutable ModifiedHuberLoss <: MarginBasedLoss end

function value{T<:Number}(loss::ModifiedHuberLoss, agreement::T)
    agreement >= -1 ? abs2(max(zero(T), one(agreement) - agreement)) : -T(4) * agreement
end
function deriv{T<:Number}(loss::ModifiedHuberLoss, agreement::T)
    if agreement >= -1
        agreement > 1 ? zero(T) : T(2)*agreement - T(2)
    else
        -T(4)
    end
end
function deriv2{T<:Number}(loss::ModifiedHuberLoss, agreement::T)
    agreement < -1 || agreement > 1 ? zero(T) : T(2)
end
value_deriv(loss::ModifiedHuberLoss, agreement::Number) = (value(loss, agreement), deriv(loss, agreement))

isdifferentiable(::ModifiedHuberLoss) = true
isdifferentiable(::ModifiedHuberLoss, at) = true
istwicedifferentiable(::ModifiedHuberLoss) = false
istwicedifferentiable(loss::ModifiedHuberLoss, at) = at != 1 && at != -1
islocallylipschitzcont(::ModifiedHuberLoss) = true
islipschitzcont(::ModifiedHuberLoss) = true
islipschitzcont_deriv(::ModifiedHuberLoss) = true
isconvex(::ModifiedHuberLoss) = true
isstronglyconvex(::ModifiedHuberLoss) = false
isclipable(::ModifiedHuberLoss) = true
