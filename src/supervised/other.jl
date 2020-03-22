@doc doc"""
    MissclassLoss

Misclassification loss that assigns `1` for misclassified
examples and `0` otherwise. It is a generalization of
`ZeroOneLoss` for more than two classes.
"""
struct MisclassLoss <: SupervisedLoss end

agreement(target, output) = target == output

value(::MisclassLoss, agreement::Bool) = agreement ? 0 : 1
deriv(::MisclassLoss, agreement::Bool) = 0
deriv2(::MisclassLoss, agreement::Bool) = 0
value_deriv(::MisclassLoss, agreement::Bool) = agreement ? (0, 0) : (1, 0)

value(loss::MisclassLoss, target::Number, output::Number) = value(loss, agreement(target, output))
deriv(loss::MisclassLoss, target::Number, output::Number) = deriv(loss, agreement(target, output))
deriv2(loss::MisclassLoss, target::Number, output::Number) = deriv2(loss, agreement(target, output))

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
function value_deriv(loss::PoissonLoss, target::Number, output::Number)
    exp_o = exp(output)
    return (exp_o-(target*output), exp_o-target)
end

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
    CrossentropyLoss <: SupervisedLoss

Cross-entropy loss also known as log loss and logistic loss is defined as:

``L(target, output) = - target*ln(output) - (1-target)*ln(1-output)``
"""
struct CrossentropyLoss <: SupervisedLoss end
const LogitProbLoss = CrossentropyLoss

function value(loss::CrossentropyLoss, target::Number, output::Number)
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
deriv(loss::CrossentropyLoss, target::Number, output::Number) = (1-target) / (1-output) - target / output
deriv2(loss::CrossentropyLoss, target::Number, output::Number) = (1-target) / (1-output)^2 + target / output^2
value_deriv(loss::CrossentropyLoss, target::Number, output::Number) = (value(loss,target,output), deriv(loss,target,output))

isdifferentiable(::CrossentropyLoss) = true
isdifferentiable(::CrossentropyLoss, y, t) = t != 0 && t != 1
istwicedifferentiable(::CrossentropyLoss) = true
istwicedifferentiable(::CrossentropyLoss, y, t) = t != 0 && t != 1
isconvex(::CrossentropyLoss) = true

# ===============================================================
# L(target, output) = sign(agreement) < 0 ? 1 : 0
