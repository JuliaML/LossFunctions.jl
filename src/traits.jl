# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module Traits

"""
Baseclass for all losses. A loss is some (possibly simplified)
function `L(x, y, ŷ)`, of features `x`, targets `y` and outputs
`ŷ = f(x)` for some function `f`.
"""
abstract type Loss end

"""
A loss is considered **supervised**, if all the information needed
to compute `L(x, ŷ, y)` are contained in `ŷ` and `y`, and thus allows
for the simplification `L(ŷ, y)`.
"""
abstract type SupervisedLoss <: Loss end

"""
A supervised loss that can be simplified to `L(ŷ, y) = L(ŷ - y)`
is considered **distance-based**.
"""
abstract type DistanceLoss <: SupervisedLoss end

"""
A supervised loss with targets `y ∈ {-1, 1}`, and which
can be simplified to `L(ŷ, y) = L(ŷ⋅y)` is considered
**margin-based**.
"""
abstract type MarginLoss <: SupervisedLoss end

"""
A loss is considered **unsupervised**, if all the information needed
to compute `L(x, ŷ, y)` are contained in `x` and `ŷ`, and thus allows
for the simplification `L(x, ŷ)`.
"""
abstract type UnsupervisedLoss <: Loss end

"""
    deriv(loss, output, target) -> Number

Compute the analytical derivative with respect to the `output` for the
`loss` function. Note that `target` and `output` can be of different
numeric type, in which case promotion is performed in the manner
appropriate for the given loss.
"""
function deriv end

"""
    deriv2(loss, output, target) -> Number

Compute the second derivative with respect to the `output` for the
`loss` function. Note that `target` and `output` can be of different
numeric type, in which case promotion is performed in the manner
appropriate for the given loss.
"""
function deriv2 end

"""
    isconvex(loss) -> Bool

Return `true` if the given `loss` denotes a convex function.
A function `f: ℝⁿ → ℝ` is convex if its domain is a convex set
and if for all `x, y` in that domain, with `θ` such that for
`0 ≦ θ ≦ 1`, we have `f(θ x + (1 - θ) y) ≦ θ f(x) + (1 - θ) f(y)`.
"""
isconvex(loss::SupervisedLoss) = isstrictlyconvex(loss)

"""
    isstrictlyconvex(loss) -> Bool

Return `true` if the given `loss` denotes a strictly convex function.
A function `f : ℝⁿ → ℝ` is strictly convex if its domain is a convex
set and if for all `x, y` in that domain where `x ≠ y`, with `θ` such
that for `0 < θ < 1`, we have `f(θ x + (1 - θ) y) < θ f(x) + (1 - θ) f(y)`.
"""
isstrictlyconvex(loss::SupervisedLoss) = isstronglyconvex(loss)

"""
    isstronglyconvex(loss) -> Bool

Return `true` if the given `loss` denotes a strongly convex function.
A function `f : ℝⁿ → ℝ` is `m`-strongly convex if its domain is a convex
set, and if for all `x, y` in that domain where `x ≠ y`, and `θ` such that
for `0 ≤ θ ≤ 1`, we have
`f(θ x + (1 - θ)y) < θ f(x) + (1 - θ) f(y) - 0.5 m ⋅ θ (1 - θ) | x - y |₂²`

In a more familiar setting, if the loss function is differentiable we have
`(∇f(x) - ∇f(y))ᵀ (x - y) ≥ m | x - y |₂²`
"""
isstronglyconvex(loss::SupervisedLoss) = false

"""
    isdifferentiable(loss, [x]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function `f : ℝⁿ → ℝᵐ` is differentiable at a point `x` in the interior
domain of `f` if there exists a matrix `Df(x) ∈ ℝ^(m × n)` such that
it satisfies:

`lim_{z ≠ x, z → x} (|f(z) - f(x) - Df(x)(z-x)|₂) / |z - x|₂ = 0`

A function is differentiable if its domain is open and it is
differentiable at every point `x`.
"""
isdifferentiable(loss::SupervisedLoss) = istwicedifferentiable(loss)
isdifferentiable(loss::SupervisedLoss, at) = isdifferentiable(loss)

"""
    istwicedifferentiable(loss, [x]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function `f : ℝⁿ → ℝ` is said to be twice differentiable
at a point `x` in the interior domain of `f`, if the function
derivative for `∇f` exists at `x`: `∇²f(x) = D∇f(x)`.

A function is twice differentiable if its domain is open and it
is twice differentiable at every point `x`.
"""
istwicedifferentiable(loss::SupervisedLoss) = false
istwicedifferentiable(loss::SupervisedLoss, at) = istwicedifferentiable(loss)

"""
    islocallylipschitzcont(loss) -> Bool

Return `true` if the given `loss` function is locally-Lipschitz
continous.

A supervised loss `L : Y × ℝ → [0, ∞)` is called locally Lipschitz
continuous if for all `a ≥ 0` there exists a constant `cₐ ≥ 0`,
such that

`sup_{y ∈ Y} | L(y,t) − L(y,t′) | ≤ cₐ |t − t′|, t, t′ ∈ [−a,a]`

Every convex function is locally lipschitz continuous.
"""
islocallylipschitzcont(loss::SupervisedLoss) = isconvex(loss) || islipschitzcont(loss)

"""
    islipschitzcont(loss) -> Bool

Return `true` if the given `loss` function is Lipschitz continuous.

A supervised loss function `L : Y × ℝ → [0, ∞)` is Lipschitz continous,
if there exists a finite constant `M < ∞` such that
`|L(y, t) - L(y, t′)| ≤ M |t - t′|, ∀ (y, t) ∈ Y × ℝ`
"""
islipschitzcont(loss::SupervisedLoss) = false

"""
    isnemitski(loss) -> Bool

Return `true` if the given `loss` denotes a Nemitski loss function.

We call a supervised loss function `L : Y × ℝ → [0,∞)` a Nemitski
loss if there exist a measurable function `b : Y → [0, ∞)` and an
increasing function `h : [0, ∞) → [0, ∞)` such that
`L(y,ŷ) ≤ b(y) + h(|ŷ|), (y, ŷ) ∈ Y × ℝ`

If a loss if locally lipsschitz continuous then it is a Nemitski loss.
"""
isnemitski(loss::SupervisedLoss) = islocallylipschitzcont(loss)
isnemitski(loss::MarginLoss) = true

"""
    isunivfishercons(loss) -> Bool
"""
isunivfishercons(loss::Loss) = false

"""
    isfishercons(loss) -> Bool

Return `true` if the givel `loss` is Fisher consistent.

We call a supervised loss function `L : Y × ℝ → [0,∞)` a Fisher
consistent loss if the population minimizer of the risk `E[L(y,f(x))]`
for all measurable functions leads to the Bayes optimal decision rule.
"""
isfishercons(loss::Loss) = isunivfishercons(loss)

"""
    isclipable(loss) -> Bool

Return `true` if the given `loss` function is clipable. A
supervised loss `L : Y × ℝ → [0,∞)` can be clipped at `M > 0`
if, for all `(y,t) ∈ Y × ℝ`, `L(y, t̂) ≤ L(y, t)` where
`t̂` denotes the clipped value of `t` at `± M`.
That is
`t̂ = -M` if `t < -M`, `t̂ = t` if `t ∈ [-M, M]`, and `t = M` if `t > M`.
"""
isclipable(loss::SupervisedLoss) = false
isclipable(loss::DistanceLoss) = true # can someone please double check?

"""
    isdistancebased(loss) -> Bool

Return `true` if the given `loss` is a distance-based loss.

A supervised loss function `L : Y × ℝ → [0,∞)` is said to be
distance-based, if there exists a representing function `ψ : ℝ → [0,∞)`
satisfying `ψ(0) = 0` and `L(y, ŷ) = ψ (ŷ - y), (y, ŷ) ∈ Y × ℝ`.
"""
isdistancebased(loss::Loss) = false
isdistancebased(loss::DistanceLoss) = true

"""
    ismarginbased(loss) -> Bool

Return `true` if the given `loss` is a margin-based loss.

A supervised loss function `L : Y × ℝ → [0,∞)` is said to be
margin-based, if there exists a representing function `ψ : ℝ → [0,∞)`
satisfying `L(y, ŷ) = ψ(y⋅ŷ), (y, ŷ) ∈ Y × ℝ`.
"""
ismarginbased(loss::Loss) = false
ismarginbased(loss::MarginLoss) = true

"""
    isclasscalibrated(loss) -> Bool
"""
isclasscalibrated(loss::SupervisedLoss) = false
isclasscalibrated(loss::MarginLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

"""
    issymmetric(loss) -> Bool

Return `true` if the given loss is a symmetric loss.

A function `f : ℝ → [0,∞)` is said to be symmetric
about origin if we have `f(x) = f(-x), ∀ x ∈ ℝ`.

A distance-based loss is said to be symmetric if its
representing function is symmetric.
"""
issymmetric(loss::SupervisedLoss) = false

"""
    isminimizable(loss) -> Bool

Return `true` if the given `loss` is a minimizable loss.
"""
isminimizable(loss::SupervisedLoss) = isconvex(loss)

export
  # export traits whenever users type:
  # using LossFunctions.Traits
  Loss,
  SupervisedLoss,
  UnsupervisedLoss,
  MarginLoss,
  DistanceLoss,
  deriv,
  deriv2,
  isdistancebased,
  ismarginbased,
  isminimizable,
  isdifferentiable,
  istwicedifferentiable,
  isconvex,
  isstrictlyconvex,
  isstronglyconvex,
  isnemitski,
  isunivfishercons,
  isfishercons,
  islipschitzcont,
  islocallylipschitzcont,
  isclipable,
  isclasscalibrated,
  issymmetric

end

import .Traits:
  # import traits into current module
  # to add method definitions
  Loss,
  SupervisedLoss,
  UnsupervisedLoss,
  MarginLoss,
  DistanceLoss,
  deriv,
  deriv2,
  isdistancebased,
  ismarginbased,
  isminimizable,
  isdifferentiable,
  istwicedifferentiable,
  isconvex,
  isstrictlyconvex,
  isstronglyconvex,
  isnemitski,
  isunivfishercons,
  isfishercons,
  islipschitzcont,
  islocallylipschitzcont,
  isclipable,
  isclasscalibrated,
  issymmetric
