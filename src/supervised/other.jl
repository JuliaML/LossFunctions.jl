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

# categorical + numerical type
const CV = Union{CategoricalValue,Number}

# specialize result type
result_type(loss::MisclassLoss{R}, t::Type{T}, o::Type{O}) where {R,T,O} = R

# return floating point to avoid big integers in aggregations
value(::MisclassLoss{R}, agreement::Bool) where R = ifelse(agreement, zero(R), one(R))
deriv(::MisclassLoss{R}, agreement::Bool) where R = zero(R)
deriv2(::MisclassLoss{R}, agreement::Bool) where R = zero(R)

value(loss::MisclassLoss, target::CV, output::CV) = value(loss, target == output)
deriv(loss::MisclassLoss, target::CV, output::CV) = deriv(loss, target == output)
deriv2(loss::MisclassLoss, target::CV, output::CV) = deriv2(loss, target == output)

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

``L(target, output) = exp(output) - target*output``
"""
struct PoissonLoss <: SupervisedLoss end

value(loss::PoissonLoss, target::Number, output::Number) = exp(output) - target*output
deriv(loss::PoissonLoss, target::Number, output::Number) = exp(output) - target
deriv2(loss::PoissonLoss, target::Number, output::Number) = exp(output)

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

Cross-entropy loss also known as log loss and logistic loss is defined as:

``L(target, output) = - target*ln(output) - (1-target)*ln(1-output)``
"""
struct CrossEntropyLoss <: SupervisedLoss end

function value(loss::CrossEntropyLoss, target::Number, output::Number)
    target >= 0 && target <=1 || error("target must be in [0,1]")
    output >= 0 && output <=1 || error("output must be in [0,1]")
    if target == 0
        -log(1 - output)
    elseif target == 1
        -log(output)
    else
        -(target * log(output) + (1-target) * log(1-output))
    end
end
deriv(loss::CrossEntropyLoss, target::Number, output::Number) = (1-target) / (1-output) - target / output
deriv2(loss::CrossEntropyLoss, target::Number, output::Number) = (1-target) / (1-output)^2 + target / output^2

isdifferentiable(::CrossEntropyLoss) = true
isdifferentiable(::CrossEntropyLoss, y, t) = t != 0 && t != 1
istwicedifferentiable(::CrossEntropyLoss) = true
istwicedifferentiable(::CrossEntropyLoss, y, t) = t != 0 && t != 1
isconvex(::CrossEntropyLoss) = true

# ===============================================================
# L(target, output) = sign(agreement) < 0 ? 1 : 0
