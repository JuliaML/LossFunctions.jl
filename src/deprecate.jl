@deprecate value!(buffer, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) buffer .= value.(loss, targets, outputs)
@deprecate deriv!(buffer, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) buffer .= deriv.(loss, targets, outputs)
@deprecate deriv2!(buffer, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray) buffer .= deriv2.(loss, targets, outputs)

@deprecate value!(buffer, loss::SupervisedLoss, array::AbstractArray) buffer .= value.(loss, array)
@deprecate deriv!(buffer, loss::SupervisedLoss, array::AbstractArray) buffer .= deriv.(loss, array)
@deprecate deriv2!(buffer, loss::SupervisedLoss, array::AbstractArray) buffer .= deriv2.(loss, array)

