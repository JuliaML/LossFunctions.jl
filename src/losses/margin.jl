# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@doc doc"""
    ZeroOneLoss <: MarginLoss

The classical classification loss. It penalizes every misclassified
observation with a loss of `1` while every correctly classified
observation has a loss of `0`.
It is not convex nor continuous and thus seldom used directly.
Instead one usually works with some classification-calibrated
surrogate loss, such as [L1HingeLoss](@ref).

```math
L(a) = \begin{cases} 1 & \quad \text{if } a < 0 \\ 0 & \quad \text{if } a >= 0\\ \end{cases}
```

---
```
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
```
"""
struct ZeroOneLoss <: MarginLoss end

deriv(loss::ZeroOneLoss, target::Number, output::Number) = zero(output)
deriv2(loss::ZeroOneLoss, target::Number, output::Number) = zero(output)

(loss::ZeroOneLoss)(agreement::T) where {T<:Number} = sign(agreement) < 0 ? one(T) : zero(T)
deriv(loss::ZeroOneLoss, agreement::T) where {T<:Number} = zero(T)
deriv2(loss::ZeroOneLoss, agreement::T) where {T<:Number} = zero(T)

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

@doc doc"""
    PerceptronLoss <: MarginLoss

The perceptron loss linearly penalizes every prediction where the
resulting `agreement <= 0`.
It is Lipschitz continuous and convex, but not strictly convex.

```math
L(a) = \max \{ 0, -a \}
```

---
```
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
```
"""
struct PerceptronLoss <: MarginLoss end

(loss::PerceptronLoss)(agreement::T) where {T<:Number} = max(zero(T), -agreement)
deriv(loss::PerceptronLoss, agreement::T) where {T<:Number} = agreement >= 0 ? zero(T) : -one(T)
deriv2(loss::PerceptronLoss, agreement::T) where {T<:Number} = zero(T)

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

@doc doc"""
    LogitMarginLoss <: MarginLoss

The margin version of the logistic loss. It is infinitely many
times differentiable, strictly convex, and Lipschitz continuous.

```math
L(a) = \ln (1 + e^{-a})
```

---
```
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
```
"""
struct LogitMarginLoss <: MarginLoss end
(loss::LogitMarginLoss)(agreement::Number) = log1p(exp(-agreement))
deriv(loss::LogitMarginLoss, agreement::Number) = -one(agreement) / (one(agreement) + exp(agreement))
deriv2(loss::LogitMarginLoss, agreement::Number) = (eᵗ = exp(agreement);
eᵗ / abs2(one(eᵗ) + eᵗ))

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

@doc doc"""
    L1HingeLoss <: MarginLoss

The hinge loss linearly penalizes every predicition where the
resulting `agreement < 1` .
It is Lipschitz continuous and convex, but not strictly convex.

```math
L(a) = \max \{ 0, 1 - a \}
```

---
```
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
```
"""
struct L1HingeLoss <: MarginLoss end
const HingeLoss = L1HingeLoss

(loss::L1HingeLoss)(agreement::T) where {T<:Number} = max(zero(T), one(T) - agreement)
deriv(loss::L1HingeLoss, agreement::T) where {T<:Number} = agreement >= 1 ? zero(T) : -one(T)
deriv2(loss::L1HingeLoss, agreement::T) where {T<:Number} = zero(T)

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

@doc doc"""
    L2HingeLoss <: MarginLoss

The truncated least squares loss quadratically penalizes every
predicition where the resulting `agreement < 1`.
It is locally Lipschitz continuous and convex,
but not strictly convex.

```math
L(a) = \max \{ 0, 1 - a \}^2
```

---
```
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
```
"""
struct L2HingeLoss <: MarginLoss end

(loss::L2HingeLoss)(agreement::T) where {T<:Number} = agreement >= 1 ? zero(T) : abs2(one(T) - agreement)
deriv(loss::L2HingeLoss, agreement::T) where {T<:Number} =
  agreement >= 1 ? zero(T) : convert(T, 2) * (agreement - one(T))
deriv2(loss::L2HingeLoss, agreement::T) where {T<:Number} = agreement >= 1 ? zero(T) : convert(T, 2)

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

@doc doc"""
    SmoothedL1HingeLoss <: MarginLoss

As the name suggests a smoothed version of the L1 hinge loss.
It is Lipschitz continuous and convex, but not strictly convex.

```math
L(a) = \begin{cases} \frac{0.5}{\gamma} \cdot \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge 1 - \gamma \\ 1 - \frac{\gamma}{2} - a & \quad \text{otherwise}\\ \end{cases}
```

---
```
              Lossfunction (γ=2)               Derivative
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
```
"""
struct SmoothedL1HingeLoss{T<:AbstractFloat} <: MarginLoss
  gamma::T

  function SmoothedL1HingeLoss{T}(γ::T) where {T}
    γ > 0 || error("γ must be strictly positive")
    new{T}(γ)
  end
end
SmoothedL1HingeLoss(γ::T) where {T<:AbstractFloat} = SmoothedL1HingeLoss{T}(γ)
SmoothedL1HingeLoss(γ) = SmoothedL1HingeLoss(Float64(γ))

function (loss::SmoothedL1HingeLoss{R})(agreement::T)::promote_type(R, T) where {R,T<:Number}
  if agreement >= 1 - loss.gamma
    R(0.5) / loss.gamma * abs2(max(zero(T), one(T) - agreement))
  else
    one(T) - loss.gamma / R(2) - agreement
  end
end
function deriv(loss::SmoothedL1HingeLoss{R}, agreement::T)::promote_type(R, T) where {R,T<:Number}
  if agreement >= 1 - loss.gamma
    agreement >= 1 ? zero(T) : (agreement - one(T)) / loss.gamma
  else
    -one(T)
  end
end
function deriv2(loss::SmoothedL1HingeLoss{R}, agreement::T)::promote_type(R, T) where {R,T<:Number}
  agreement < 1 - loss.gamma || agreement > 1 ? zero(T) : one(T) / loss.gamma
end

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

@doc doc"""
    ModifiedHuberLoss <: MarginLoss

A special (4 times scaled) case of the [`SmoothedL1HingeLoss`](@ref)
with `γ=2`. It is Lipschitz continuous and convex,
but not strictly convex.

```math
L(a) = \begin{cases} \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge -1 \\ - 4 a & \quad \text{otherwise}\\ \end{cases}
```

---
```
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
```
"""
struct ModifiedHuberLoss <: MarginLoss end

function (loss::ModifiedHuberLoss)(agreement::T) where {T<:Number}
  agreement >= -1 ? abs2(max(zero(T), one(agreement) - agreement)) : -convert(T, 4) * agreement
end
function deriv(loss::ModifiedHuberLoss, agreement::T) where {T<:Number}
  if agreement >= -1
    agreement > 1 ? zero(T) : convert(T, 2) * agreement - convert(T, 2)
  else
    -convert(T, 4)
  end
end
function deriv2(loss::ModifiedHuberLoss, agreement::T) where {T<:Number}
  agreement < -1 || agreement > 1 ? zero(T) : convert(T, 2)
end

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

@doc doc"""
    L2MarginLoss <: MarginLoss

The margin-based least-squares loss for classification,
which penalizes every prediction where `agreement != 1` quadratically.
It is locally Lipschitz continuous and strongly convex.

```math
L(a) = {\left( 1 - a \right)}^2
```

---
```
              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    5 │     .                   │    2 │                       ,r│
      │     '.                  │      │                     ,/  │
      │      '\                 │      │                   ,/    │
      │        \                │      ├                 ,/      ┤
    L │         '.              │   L' │               ./        │
      │          '.             │      │             ./          │
      │            \.          .│      │           ./            │
    0 │              '-.____.-' │   -3 │         ./              │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
```
"""
struct L2MarginLoss <: MarginLoss end

(loss::L2MarginLoss)(agreement::T) where {T<:Number} = abs2(one(T) - agreement)
deriv(loss::L2MarginLoss, agreement::T) where {T<:Number} = convert(T, 2) * (agreement - one(T))
deriv2(loss::L2MarginLoss, agreement::T) where {T<:Number} = convert(T, 2)

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

# ============================================================

@doc doc"""
    ExpLoss <: MarginLoss

The margin-based exponential loss for classification, which
penalizes every prediction exponentially. It is infinitely many
times differentiable, locally Lipschitz continuous and strictly
convex, but not clipable.

```math
L(a) = e^{-a}
```

---
```
              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    5 │  \.                     │    0 │               _,,---:'""│
      │   l                     │      │           _r/"'         │
      │    l.                   │      │        .r/'             │
      │     ":                  │      │      .r'                │
    L │       \.                │   L' │     ./                  │
      │        "\..             │      │    .'                   │
      │           '":,_         │      │   ,'                    │
    0 │                ""---:.__│   -5 │  ./                     │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
```
"""
struct ExpLoss <: MarginLoss end

(loss::ExpLoss)(agreement::Number) = exp(-agreement)
deriv(loss::ExpLoss, agreement::Number) = -exp(-agreement)
deriv2(loss::ExpLoss, agreement::Number) = exp(-agreement)

isunivfishercons(::ExpLoss) = true
isdifferentiable(::ExpLoss) = true
isdifferentiable(::ExpLoss, at) = true
istwicedifferentiable(::ExpLoss) = true
istwicedifferentiable(::ExpLoss, at) = true
islocallylipschitzcont(::ExpLoss) = true
islipschitzcont(::ExpLoss) = false
isconvex(::ExpLoss) = true
isstrictlyconvex(::ExpLoss) = true
isstronglyconvex(::ExpLoss) = false
isclipable(::ExpLoss) = false

# ============================================================

@doc doc"""
    SigmoidLoss <: MarginLoss

Continuous loss which penalizes every prediction with a loss
within in the range (0,2). It is infinitely many times
differentiable, Lipschitz continuous but nonconvex.

```math
L(a) = 1 - \tanh(a)
```

---
```
              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │""'--,.                  │    0 │..                     ..│
      │      '\.                │      │ "\.                 ./" │
      │         '.              │      │    ',             ,'    │
      │           \.            │      │      \           /      │
    L │            "\.          │   L' │       \         /       │
      │              \.         │      │        \.     ./        │
      │                \,       │      │         \.   ./         │
    0 │                  '"-:.__│   -1 │          ',_,'          │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
```
"""
struct SigmoidLoss <: MarginLoss end

(loss::SigmoidLoss)(agreement::Number) = one(agreement) - tanh(agreement)
deriv(loss::SigmoidLoss, agreement::Number) = -abs2(sech(agreement))
deriv2(loss::SigmoidLoss, agreement::T) where {T<:Number} = convert(T, 2) * tanh(agreement) * abs2(sech(agreement))

isunivfishercons(::SigmoidLoss) = true
isdifferentiable(::SigmoidLoss) = true
isdifferentiable(::SigmoidLoss, at) = true
istwicedifferentiable(::SigmoidLoss) = true
istwicedifferentiable(::SigmoidLoss, at) = true
islocallylipschitzcont(::SigmoidLoss) = true
islipschitzcont(::SigmoidLoss) = true
isclasscalibrated(::SigmoidLoss) = true
isconvex(::SigmoidLoss) = false
isstrictlyconvex(::SigmoidLoss) = false
isstronglyconvex(::SigmoidLoss) = false
isclipable(::SigmoidLoss) = false

# ============================================================

@doc doc"""
    DWDMarginLoss <: MarginLoss

The distance weighted discrimination margin loss. It is a
differentiable generalization of the [L1HingeLoss](@ref) that is
different than the [SmoothedL1HingeLoss](@ref). It is Lipschitz
continuous and convex, but not strictly convex.

```math
L(a) = \begin{cases} 1 - a & \quad \text{if } a \ge \frac{q}{q+1} \\ \frac{1}{a^q} \frac{q^q}{(q+1)^{q+1}} & \quad \text{otherwise}\\ \end{cases}
```

---
```
              Lossfunction (q=1)               Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    2 │      ".                 │    0 │                     ._r-│
      │        \.               │      │                   ./    │
      │         ',              │      │                 ./      │
      │           \.            │      │                 /       │
    L │            "\.          │   L' │                .        │
      │              \.         │      │                /        │
      │               ":__      │      │               ;         │
    0 │                   '""---│   -1 │---------------┘         │
      └────────────┴────────────┘      └────────────┴────────────┘
      -2                        2      -2                        2
                 y ⋅ ŷ                            y ⋅ ŷ
```
"""
struct DWDMarginLoss{T<:AbstractFloat} <: MarginLoss
  q::T
  function DWDMarginLoss{T}(q::T) where {T}
    q > 0 || error("q must be strictly positive")
    new{T}(q)
  end
end
DWDMarginLoss(q::T) where {T<:AbstractFloat} = DWDMarginLoss{T}(q)
DWDMarginLoss(q) = DWDMarginLoss(Float64(q))

function (loss::DWDMarginLoss{R})(agreement::T)::promote_type(R, T) where {R,T<:Number}
  q = loss.q
  if agreement <= q / (q + 1)
    R(1) - agreement
  else
    (q^q / (q + 1)^(q + 1)) / agreement^q
  end
end

function deriv(loss::DWDMarginLoss{R}, agreement::T)::promote_type(R, T) where {R,T<:Number}
  q = loss.q
  agreement <= q / (q + 1) ? -one(T) : -(q / (q + 1))^(q + 1) / agreement^(q + 1)
end

function deriv2(loss::DWDMarginLoss{R}, agreement::T)::promote_type(R, T) where {R,T<:Number}
  q = loss.q
  agreement <= q / (q + 1) ? zero(T) : ((q^(q + 1)) / ((q + 1)^q)) / agreement^(q + 2)
end

isdifferentiable(::DWDMarginLoss) = true
isdifferentiable(::DWDMarginLoss, at) = true
istwicedifferentiable(::DWDMarginLoss) = true
istwicedifferentiable(loss::DWDMarginLoss, at) = true
islocallylipschitzcont(::DWDMarginLoss) = true
islipschitzcont(::DWDMarginLoss) = true
isconvex(::DWDMarginLoss) = true
isstrictlyconvex(::DWDMarginLoss) = false
isstronglyconvex(::DWDMarginLoss) = false
isfishercons(::DWDMarginLoss) = true
isunivfishercons(::DWDMarginLoss) = true
isclipable(::DWDMarginLoss) = false
