# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@doc doc"""
    MisclassLoss{R<:AbstractFloat} <: SupervisedLoss

Misclassification loss that assigns `1` for misclassified
examples and `0` otherwise. It is a generalization of
`ZeroOneLoss` for more than two classes.

The type parameter `R` specifies the result type of the
loss. Default type is double precision `R = Float64`.
"""
struct MisclassLoss{R<:AbstractFloat} <: SupervisedLoss end

MisclassLoss() = MisclassLoss{Float64}()

# return floating point to avoid big integers in aggregations
(::MisclassLoss{R})(agreement::Bool) where {R} = ifelse(agreement, zero(R), one(R))
deriv(::MisclassLoss{R}, agreement::Bool) where {R} = zero(R)
deriv2(::MisclassLoss{R}, agreement::Bool) where {R} = zero(R)

(loss::MisclassLoss)(output::Number, target::Number) = loss(target == output)
deriv(loss::MisclassLoss, output::Number, target::Number) = deriv(loss, target == output)
deriv2(loss::MisclassLoss, output::Number, target::Number) = deriv2(loss, target == output)

isminimizable(::MisclassLoss) = false
isdifferentiable(::MisclassLoss) = false
isdifferentiable(::MisclassLoss, at) = at != 0
istwicedifferentiable(::MisclassLoss) = false
istwicedifferentiable(::MisclassLoss, at) = at != 0
isnemitski(::MisclassLoss) = false
islipschitzcont(::MisclassLoss) = false
isconvex(::MisclassLoss) = false
isclasscalibrated(::MisclassLoss) = false
isclipable(::MisclassLoss) = false

# ===============================================================

@doc doc"""
    PoissonLoss <: SupervisedLoss

Loss under a Poisson noise distribution (KL-divergence)

``L(output, target) = exp(output) - target*output``
"""
struct PoissonLoss <: SupervisedLoss end

(loss::PoissonLoss)(output::Number, target::Number) = exp(output) - target * output
deriv(loss::PoissonLoss, output::Number, target::Number) = exp(output) - target
deriv2(loss::PoissonLoss, output::Number, target::Number) = exp(output)

isdifferentiable(::PoissonLoss) = true
isdifferentiable(::PoissonLoss, y, t) = true
istwicedifferentiable(::PoissonLoss) = true
istwicedifferentiable(::PoissonLoss, y, t) = true
islipschitzcont(::PoissonLoss) = false
isconvex(::PoissonLoss) = true
# TODO: isstrictlyconvex(::PoissonLoss) = ?
isstronglyconvex(::PoissonLoss) = false

# ===============================================================

@doc doc"""
    CrossEntropyLoss <: SupervisedLoss

The cross-entropy loss is defined as:

``L(output, target) = - target*log(output) - (1-target)*log(1-output)``
"""
struct CrossEntropyLoss <: SupervisedLoss end

function (loss::CrossEntropyLoss)(output::Number, target::Number)
  target >= 0 && target <= 1 || error("target must be in [0,1]")
  output >= 0 && output <= 1 || error("output must be in [0,1]")
  if target == 0
    -log(1 - output)
  elseif target == 1
    -log(output)
  else
    -(target * log(output) + (1 - target) * log(1 - output))
  end
end
deriv(loss::CrossEntropyLoss, output::Number, target::Number) = (1 - target) / (1 - output) - target / output
deriv2(loss::CrossEntropyLoss, output::Number, target::Number) = (1 - target) / (1 - output)^2 + target / output^2

isdifferentiable(::CrossEntropyLoss) = true
isdifferentiable(::CrossEntropyLoss, y, t) = t != 0 && t != 1
istwicedifferentiable(::CrossEntropyLoss) = true
istwicedifferentiable(::CrossEntropyLoss, y, t) = t != 0 && t != 1
isconvex(::CrossEntropyLoss) = true
