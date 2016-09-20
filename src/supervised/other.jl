# ===============================================================
# L(y, t) = exp(t) - t*y

"""
    PoissonLoss <: SupervisedLoss

Loss under a Poisson noise distribution (KL-divergence)
"""
immutable PoissonLoss <: SupervisedLoss end

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
islipschitzcont_deriv(::PoissonLoss) = false
isconvex(::PoissonLoss) = true
# TODO: isstrictlyconvex(::PoissonLoss) = ?
isstronglyconvex(::PoissonLoss) = false

# ===============================================================
# L(target, output) = - target*ln(output) - (1-target)*ln(1-output)

immutable CrossentropyLoss <: SupervisedLoss end
typealias LogitProbLoss CrossentropyLoss

function value(loss::CrossentropyLoss, target::Number, output::Number)
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
isconvex(::CrossentropyLoss) = true

# ===============================================================
# L(target, output) = sign(agreement) < 0 ? 1 : 0

