# Note: agreement = output * target
#       Agreement is high when output and target are the same sign and |output| is large.
#       It is an indication that the output represents the correct class in a margin-based model.

# ============================================================
# L(target, output) = max(0, -agreement)

"""
    PerceptronLoss <: MarginLoss

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │\\.                       │    0 │            ┌------------│
      │ '..                     │      │            |            │
      │   \\.                    │      │            |            │
      │     '.                  │      │            |            │
      │      '.                 │      │            |            │
      │        \\.               │      │            |            │
      │         '.              │      │            |            │
    0 │           \\.____________│   -1 │------------┘            │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                y * h(x)                         y * h(x)
"""
immutable PerceptronLoss <: MarginLoss end

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

# ============================================================
# L(target, output) = ln(1 + exp(-agreement))

"""
    LogitMarginLoss <: MarginLoss

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │ \\.                      │    0 │                  ._--/""│
      │   \\.                    │      │               ../'      │
      │     \\.                  │      │              ./         │
      │       \\..               │      │            ./'          │
      │         '-_             │      │          .,'            │
      │            '-_          │      │         ./              │
      │               '\\-._     │      │      .,/'               │
    0 │                    '""*-│   -1 │__.--''                  │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -4                        4
                y * h(x)                         y * h(x)
"""
immutable LogitMarginLoss <: MarginLoss end
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

# ============================================================
# L(target, output) = max(0, 1 - agreement)

# lineplot(HingeLoss(), canvas = AsciiCanvas, width = 25, height = 8, xlim=[-2,2], ylim=[0,3])
# lineplot(deriv_fun(HingeLoss()), 0:.01:2, canvas = AsciiCanvas, width = 25, height = 8, xlim=[0,2], ylim=[-1,0])

"""
    L1HingeLoss <: MarginLoss

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    3 │'\\.                      │    0 │                  ┌------│
      │  ''_                    │      │                  |      │
      │     \\.                  │      │                  |      │
      │       '.                │      │                  |      │
      │         ''_             │      │                  |      │
      │            \\.           │      │                  |      │
      │              '.         │      │                  |      │
    0 │                ''_______│   -1 │------------------┘      │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                y * h(x)                         y * h(x)
"""
immutable L1HingeLoss <: MarginLoss end
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

# ============================================================
# L(target, output) = max(0, 1 - agreement)^2

"""
    L2HingeLoss <: MarginLoss

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    5 │     .                   │    0 │                 ,r------│
      │     '.                  │      │               ,/        │
      │      '\\                 │      │             ,/          │
      │        \\                │      │           ,/            │
      │         '.              │      │         ./              │
      │          '.             │      │       ./                │
      │            \\.           │      │     ./                  │
    0 │              '-.________│   -5 │   ./                    │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                y * h(x)                         y * h(x)
"""
immutable L2HingeLoss <: MarginLoss end

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

# ============================================================
# L(target, output) = 0.5 / γ * max(0, 1 - agreement)^2   ... agreement >= 1 - γ
#                     1 - γ / 2 - agreement               ... otherwise

"""
    SmoothedL1HingeLoss <: MarginLoss

              Lossfunction (γ=1)               Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │\\.                       │    0 │                 ,r------│
      │ '.                      │      │               ./'       │
      │   \\.                    │      │              ,/         │
      │     '.                  │      │            ./'          │
      │      '.                 │      │           ,'            │
      │        \\.               │      │         ,/              │
      │          ',             │      │       ./'               │
    0 │            '*-._________│   -1 │______./                 │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                y * h(x)                         y * h(x)
"""
immutable SmoothedL1HingeLoss <: MarginLoss
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

# ============================================================
# L(target, output) = max(0, 1 - agreement)^2    ... agreement >= -1
#                     -4*agreement               ... otherwise

"""
    ModifiedHuberLoss <: MarginLoss

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    5 │    '.                   │    0 │                .+-------│
      │     '.                  │      │              ./'        │
      │      '\\                 │      │             ,/          │
      │        \\                │      │           ,/            │
      │         '.              │      │         ./              │
      │          '.             │      │       ./'               │
      │            \\.           │      │______/'                 │
    0 │              '-.________│   -5 │                         │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                y * h(x)                         y * h(x)
"""
immutable ModifiedHuberLoss <: MarginLoss end

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

