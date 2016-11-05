# Note: agreement = output * target
#       Agreement is high when output and target are the same sign and |output| is large.
#       It is an indication that the output represents the correct class in a margin-based model.

# ============================================================

doc"""
    ZeroOneLoss <: MarginLoss

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    1 │------------┐            │    1 │                         │
      │            |            │      │                         │
      │            |            │      │                         │
      │            |            │      │_________________________│
      │            |            │      │                         │
      │            |            │      │                         │
      │            |            │      │                         │
    0 │            └------------│   -1 │                         │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                y * h(x)                         y * h(x)
"""
immutable ZeroOneLoss <: MarginLoss end

deriv(loss::ZeroOneLoss, target::Number, output::Number) = zero(output)
deriv2(loss::ZeroOneLoss, target::Number, output::Number) = zero(output)

value{T<:Number}(loss::ZeroOneLoss, agreement::T) = sign(agreement) < 0 ? one(T) : zero(T)
deriv{T<:Number}(loss::ZeroOneLoss, agreement::T) = zero(T)
deriv2{T<:Number}(loss::ZeroOneLoss, agreement::T) = zero(T)
value_deriv{T<:Number}(loss::ZeroOneLoss, agreement::T) = sign(agreement) < 0 ? (one(T), zero(T)) : (zero(T), zero(T))

isminimizable(::ZeroOneLoss) = true
isdifferentiable(::ZeroOneLoss) = false
isdifferentiable(::ZeroOneLoss, at) = at != 0
istwicedifferentiable(::ZeroOneLoss) = false
istwicedifferentiable(::ZeroOneLoss, at) = at != 0
isnemitski(::ZeroOneLoss) = true
islipschitzcont(::ZeroOneLoss) = true
isconvex(::ZeroOneLoss) = false
isclasscalibrated(loss::ZeroOneLoss) = true
isclipable(::ZeroOneLoss) = true

# ============================================================

doc"""
    PerceptronLoss <: MarginLoss

The perceptron loss linearly penalizes every prediction where the
resulting `agreement <= 0`.
It is Lipshitz continuous and convex, but not strictly convex.

$L(y, ŷ) = max(0, -y⋅ŷ)$

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │\.                       │    0 │            ┌------------│
      │ '..                     │      │            |            │
      │   \.                    │      │            |            │
      │     '.                  │      │            |            │
    L │      '.                 │   L' │            |            │
      │        \.               │      │            |            │
      │         '.              │      │            |            │
    0 │           \.____________│   -1 │------------┘            │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
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
isconvex(::PerceptronLoss) = true
isstrictlyconvex(::PerceptronLoss) = false
isstronglyconvex(::PerceptronLoss) = false
isclipable(::PerceptronLoss) = true

# ============================================================

doc"""
    LogitMarginLoss <: MarginLoss

The margin version of the logistic loss. It is infinitely many
times differentiable, strictly convex, and lipschitz continuous.

$L(y, ŷ) = ln(1 + exp(-y⋅ŷ))$

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │ \.                      │    0 │                  ._--/""│
      │   \.                    │      │               ../'      │
      │     \.                  │      │              ./         │
      │       \..               │      │            ./'          │
    L │         '-_             │   L' │          .,'            │
      │            '-_          │      │         ./              │
      │               '\-._     │      │      .,/'               │
    0 │                    '""*-│   -1 │__.--''                  │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -4                        4
                 y ⋅ ŷ                            y ⋅ ŷ
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
isconvex(::LogitMarginLoss) = true
isstrictlyconvex(::LogitMarginLoss) = true
isstronglyconvex(::LogitMarginLoss) = false
isclipable(::LogitMarginLoss) = false

# ============================================================

doc"""
    L1HingeLoss <: MarginLoss

The hinge loss linearly penalizes every predicition where the
resulting `agreement <= 1` .
It is Lipshitz continuous and convex, but not strictly convex.

$L(y, ŷ) = max(0, 1 - y⋅ŷ)$

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    3 │'\.                      │    0 │                  ┌------│
      │  ''_                    │      │                  |      │
      │     \.                  │      │                  |      │
      │       '.                │      │                  |      │
    L │         ''_             │   L' │                  |      │
      │            \.           │      │                  |      │
      │              '.         │      │                  |      │
    0 │                ''_______│   -1 │------------------┘      │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
"""
immutable L1HingeLoss <: MarginLoss end
typealias HingeLoss L1HingeLoss

value{T<:Number}(loss::L1HingeLoss, agreement::T) = max(zero(T), one(T) - agreement)
deriv{T<:Number}(loss::L1HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : -one(T)
deriv2{T<:Number}(loss::L1HingeLoss, agreement::T) = zero(T)
value_deriv{T<:Number}(loss::L1HingeLoss, agreement::T) = agreement >= 1 ? (zero(T), zero(T)) : (one(T) - agreement, -one(T))

isfishercons(::L1HingeLoss) = true
isdifferentiable(::L1HingeLoss) = false
isdifferentiable(::L1HingeLoss, at) = at != 1
istwicedifferentiable(::L1HingeLoss) = false
istwicedifferentiable(::L1HingeLoss, at) = at != 1
islipschitzcont(::L1HingeLoss) = true
isconvex(::L1HingeLoss) = true
isstrictlyconvex(::L1HingeLoss) = false
isstronglyconvex(::L1HingeLoss) = false
isclipable(::L1HingeLoss) = true

# ============================================================

doc"""
    L2HingeLoss <: MarginLoss

The truncated least squares loss quadratically penalizes every
predicition where the resulting `agreement <= 1`.
It is locally Lipshitz continuous and convex, but not strictly convex.

$L(y, ŷ) = max(0, 1 - y⋅ŷ)²$

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    5 │     .                   │    0 │                 ,r------│
      │     '.                  │      │               ,/        │
      │      '\                 │      │             ,/          │
      │        \                │      │           ,/            │
    L │         '.              │   L' │         ./              │
      │          '.             │      │       ./                │
      │            \.           │      │     ./                  │
    0 │              '-.________│   -5 │   ./                    │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
"""
immutable L2HingeLoss <: MarginLoss end

value{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : abs2(one(T) - agreement)
deriv{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : T(2) * (agreement - one(T))
deriv2{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? zero(T) : T(2)
value_deriv{T<:Number}(loss::L2HingeLoss, agreement::T) = agreement >= 1 ? (zero(T), zero(T)) : (abs2(one(T) - agreement), T(2) * (agreement - one(T)))

isunivfishercons(::L2HingeLoss) = true
isdifferentiable(::L2HingeLoss) = true
isdifferentiable(::L2HingeLoss, at) = true
istwicedifferentiable(::L2HingeLoss) = false
istwicedifferentiable(::L2HingeLoss, at) = at != 1
islocallylipschitzcont(::L2HingeLoss) = true
islipschitzcont(::L2HingeLoss) = false
isconvex(::L2HingeLoss) = true
isstrictlyconvex(::L2HingeLoss) = false
isstronglyconvex(::L2HingeLoss) = false
isclipable(::L2HingeLoss) = true

# ============================================================

doc"""
    SmoothedL1HingeLoss <: MarginLoss

As the name suggests a smoothed version of the L1 hinge loss.
It is Lipshitz continuous and convex, but not strictly convex.

$L(y, ŷ) = 0.5 / γ * max(0, 1 - y⋅ŷ)²    ... y⋅ŷ ≥ 1 - γ$
$L(y, ŷ) = 1 - γ / 2 - y⋅ŷ               ... otherwise$

              Lossfunction (γ=1)               Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │\.                       │    0 │                 ,r------│
      │ '.                      │      │               ./'       │
      │   \.                    │      │              ,/         │
      │     '.                  │      │            ./'          │
    L │      '.                 │   L' │           ,'            │
      │        \.               │      │         ,/              │
      │          ',             │      │       ./'               │
    0 │            '*-._________│   -1 │______./                 │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
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
isconvex(::SmoothedL1HingeLoss) = true
isstrictlyconvex(::SmoothedL1HingeLoss) = false
isstronglyconvex(::SmoothedL1HingeLoss) = false
isclipable(::SmoothedL1HingeLoss) = true

# ============================================================

doc"""
    ModifiedHuberLoss <: MarginLoss

A special (scaled) case of the `SmoothedL1HingeLoss` with `γ=4`.
It is Lipshitz continuous and convex, but not strictly convex.

$L(y, ŷ) = max(0, 1 - y⋅ŷ)^2    ... y⋅ŷ >= -1$
$L(y, ŷ) = -4⋅y⋅ŷ               ... otherwise$

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    5 │    '.                   │    0 │                .+-------│
      │     '.                  │      │              ./'        │
      │      '\                 │      │             ,/          │
      │        \                │      │           ,/            │
    L │         '.              │   L' │         ./              │
      │          '.             │      │       ./'               │
      │            \.           │      │______/'                 │
    0 │              '-.________│   -5 │                         │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
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
isconvex(::ModifiedHuberLoss) = true
isstrictlyconvex(::ModifiedHuberLoss) = false
isstronglyconvex(::ModifiedHuberLoss) = false
isclipable(::ModifiedHuberLoss) = true

# ============================================================

doc"""
    L2MarginLoss <: MarginLoss

Margin-based least-squares loss.
It is locally Lipschitz continuous and strictly convex.

$L(y, ŷ) = (1 - y⋅ŷ)^2$
"""
immutable L2MarginLoss <: MarginLoss end

value{T<:Number}(loss::L2MarginLoss, agreement::T) = abs2(one(T) - agreement)
deriv{T<:Number}(loss::L2MarginLoss, agreement::T) = T(2) * (agreement - one(T))
deriv2{T<:Number}(loss::L2MarginLoss, agreement::T) = T(2)
value_deriv{T<:Number}(loss::L2MarginLoss, agreement::T) = (abs2(one(T) - agreement), T(2) * (agreement - one(T)))

isunivfishercons(::L2MarginLoss) = true
isdifferentiable(::L2MarginLoss) = true
isdifferentiable(::L2MarginLoss, at) = true
istwicedifferentiable(::L2MarginLoss) = true
istwicedifferentiable(::L2MarginLoss, at) = true
islocallylipschitzcont(::L2MarginLoss) = true
islipschitzcont(::L2MarginLoss) = false
isconvex(::L2MarginLoss) = true
isstrictlyconvex(::L2MarginLoss) = true
isstronglyconvex(::L2MarginLoss) = true
isclipable(::L2MarginLoss) = true

