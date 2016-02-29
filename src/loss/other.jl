
# ==========================================================================
# L(target, output) = - target*ln(output) - (1-target)*ln(1-output)

immutable CrossentropyLoss <: PredictionLoss end
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

# ==========================================================================
# L(target, output) = sign(yt) < 0 ? 1 : 0

immutable ZeroOneLoss <: PredictionLoss end

value(loss::ZeroOneLoss, target::Number, output::Number) = value(loss, target * output)
deriv(loss::ZeroOneLoss, target::Number, output::Number) = zero(output)
deriv2(loss::ZeroOneLoss, target::Number, output::Number) = zero(output)

value{T<:Number}(loss::ZeroOneLoss, yt::T) = sign(yt) < 0 ? one(T) : zero(T)
deriv{T<:Number}(loss::ZeroOneLoss, yt::T) = zero(T)
deriv2{T<:Number}(loss::ZeroOneLoss, yt::T) = zero(T)

isdifferentiable(::ZeroOneLoss) = false
isconvex(::ZeroOneLoss) = false
isclasscalibrated(loss::ZeroOneLoss) = true
