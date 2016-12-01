@deprecate value(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) value.(loss, targets, outputs)
@deprecate deriv(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) deriv.(loss, targets, outputs)
@deprecate deriv2(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) deriv2.(loss, targets, outputs)

@deprecate value!(buffer, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) buffer .= value.(loss, targets, outputs)
@deprecate deriv!(buffer, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) buffer .= deriv.(loss, targets, outputs)
@deprecate deriv2!(buffer, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) buffer .= deriv2.(loss, targets, outputs)

@deprecate value(loss::SupervisedLoss, array::AbstractArray) value.(loss, array)
@deprecate deriv(loss::SupervisedLoss, array::AbstractArray) deriv.(loss, array)
@deprecate deriv2(loss::SupervisedLoss, array::AbstractArray) deriv2.(loss, array)

@deprecate value!(buffer, loss::SupervisedLoss, array::AbstractArray) buffer .= value.(loss, array)
@deprecate deriv!(buffer, loss::SupervisedLoss, array::AbstractArray) buffer .= deriv.(loss, array)
@deprecate deriv2!(buffer, loss::SupervisedLoss, array::AbstractArray) buffer .= deriv2.(loss, array)

