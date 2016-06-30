@inline function value_fun(c::SupervisedLoss)
    _value(args...) = value(c, args...)
    _value
end

@inline function deriv_fun(c::SupervisedLoss)
    _deriv(args...) = deriv(c, args...)
    _deriv
end

@inline function deriv2_fun(c::SupervisedLoss)
    _deriv2(args...) = deriv2(c, args...)
    _deriv2
end

@inline function value_deriv_fun(c::SupervisedLoss)
    _value_deriv(args...) = value_deriv(c, args...)
    _value_deriv
end

isminimizable(c::SupervisedLoss) = isconvex(c)
isdifferentiable(c::SupervisedLoss) = istwicedifferentiable(c)
istwicedifferentiable(::SupervisedLoss) = false
isdifferentiable(c::SupervisedLoss, at) = isdifferentiable(c)
istwicedifferentiable(c::SupervisedLoss, at) = istwicedifferentiable(c)
isconvex(::SupervisedLoss) = false
isstronglyconvex(::SupervisedLoss) = false

# ===============================================================

isnemitski(loss::SupervisedLoss) = islocallylipschitzcont(loss)
islipschitzcont(::SupervisedLoss) = false
islocallylipschitzcont(::SupervisedLoss) = false
isclipable(::SupervisedLoss) = false
islipschitzcont_deriv(::SupervisedLoss) = false

# --------------------------------------------------------------

@inline function value(loss::SupervisedLoss, target::AbstractVecOrMat, output::AbstractVecOrMat)
    buffer = similar(output)
    value!(buffer, loss, target, output)
end

@inline function deriv(loss::SupervisedLoss, target::AbstractVector, output::AbstractVecOrMat)
    buffer = similar(output)
    deriv!(buffer, loss, target, output)
end

@inline function grad(loss::SupervisedLoss, target::AbstractMatrix, output::AbstractVecOrMat)
    buffer = similar(output)
    grad!(buffer, loss, target, output)
end

# --------------------------------------------------------------

function value!(buffer::AbstractVector, loss::SupervisedLoss, target::AbstractVector, output::AbstractVector)
    n = length(output)
    @_dimcheck length(target) == n && size(buffer) == size(output)
    @simd for i = 1:n
        @inbounds buffer[i] = value(loss, target[i], output[i])
    end
    buffer
end

function deriv!(buffer::AbstractVector, loss::SupervisedLoss, target::AbstractVector, output::AbstractVector)
    n = length(output)
    @_dimcheck length(target) == n && size(buffer) == size(output)
    @simd for i = 1:n
        @inbounds buffer[i] = deriv(loss, target[i], output[i])
    end
    buffer
end

# --------------------------------------------------------------

function value!(buffer::AbstractMatrix, loss::SupervisedLoss, target::AbstractVector, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck length(target) == n && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = value(loss, target[i], output[j, i])
        end
    end
    buffer
end

function deriv!(buffer::AbstractMatrix, loss::SupervisedLoss, target::AbstractVector, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck length(target) == n && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = deriv(loss, target[i], output[j, i])
        end
    end
    buffer
end

# --------------------------------------------------------------

function value!(buffer::AbstractMatrix, loss::SupervisedLoss, target::AbstractMatrix, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck size(target) == size(output) && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = value(loss, target[j, i], output[j, i])
        end
    end
    buffer
end

function grad!(buffer::AbstractMatrix, loss::SupervisedLoss, target::AbstractMatrix, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck size(target) == size(output) && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = deriv(loss, target[j, i], output[j, i])
        end
    end
    buffer
end

# --------------------------------------------------------------

function sumvalue{T<:Number}(loss::SupervisedLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += value(loss, target[i], output[i])
    end
    val
end

function sumderiv{T<:Number}(loss::SupervisedLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += deriv(loss, target[i], output[i])
    end
    val
end

# --------------------------------------------------------------

function meanvalue{T<:Number}(loss::SupervisedLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    tmp = zero(T)
    @simd for i = 1:n
        @inbounds tmp = value(loss, target[i], output[i])::T
        tmp /= n
        val += tmp
    end
    val
end

function meanderiv{T<:Number}(loss::SupervisedLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    tmp = zero(T)
    @simd for i = 1:n
        @inbounds tmp = deriv(loss, target[i], output[i])::T
        tmp /= n
        val += tmp
    end
    val
end

ismarginbased(::SupervisedLoss) = false
isclasscalibrated(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false

# ==============================================================

# abstract MarginLoss <: SupervisedLoss

@inline value(loss::MarginLoss, target::Number, output::Number) = value(loss, target * output)
@inline deriv(loss::MarginLoss, target::Number, output::Number) = target * deriv(loss, target * output)
@inline deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)
@inline function value_deriv(loss::MarginLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end

isunivfishercons(::MarginLoss) = false
isfishercons(loss::MarginLoss) = isunivfishercons(loss)
isnemitski(::MarginLoss) = true
islocallylipschitzcont(loss::MarginLoss) = isconvex(loss)
ismarginbased(::MarginLoss) = true
isclasscalibrated(loss::MarginLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

# ==============================================================

# abstract DistanceLoss <: SupervisedLoss

@inline value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
@inline deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
@inline deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
@inline value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false

